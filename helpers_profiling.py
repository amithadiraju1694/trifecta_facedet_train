from typing import Tuple
import torch
from fvcore.nn.jit_handles import get_shape
from fvcore.nn import FlopCountAnalysis


def count_parameters(model) -> Tuple[int, float, dict]:

    """
    Counts parameters of model , along with memory foot print of params and
    how many trainable parameters are available by module.

    Args:
        model
    
    Returns:
        1) Sum of total trainable params, interger.
        2) Memory in mega bytes occupied by trainable parameters.
        3) Trainable parameters dictionary by module.
    """

    trainable_params = []
    trainable_params_by_mod ={}
    trainable_params_mem_mb = []
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param.numel())
            trainable_params_by_mod[param_name] = param.numel()
            trainable_params_mem_mb.append(
                ( param.numel() * param.element_size() ) / (1024**2)
            )

    return sum(trainable_params), sum(trainable_params_mem_mb), trainable_params_by_mod


def get_all_inline_ops(model, example_input):
    """
    Function that identifies all opeartions in a model's computation graph including unsupported ones and
    returns a list of those op names. 

    Note: It's ok to have some torch operations inside regular python functions, but they must be called from a class which has nn.Module at class definition.

    Args:
        model
        example_input
    Returns:
        ops -> List of operation names in string that can be used to compute FLOPS of specific operations.
    """
    
    gm = torch.jit.trace(model.eval(), example_input)
    ops = sorted({n.kind() for n in gm.inlined_graph.nodes()})
    # Removing certain primitive ops
    ops = [op for op in ops if not op.startswith('prim::')]
    return ops




# Unary/elemwise helpers (assumes _numel(shape) is already defined)
def handle_elemwise(inputs, outputs):          # e.g., aten::add, mul
    return _numel(get_shape(outputs[0]))


def _numel(shape):
    n = 1
    for s in shape:
        n *= int(s) if s is not None else 1
    return n

def handle_sum(inputs, outputs):          # aten::sum (reduction)
    return _numel(get_shape(inputs[0]))   # ~ one add per input element

def handle_mean(inputs, outputs):         # aten::mean (reduction)
    return _numel(get_shape(inputs[0]))   # sum + scale ≈ N

def handle_std(inputs, outputs):          # aten::std (reduction)
    nin  = _numel(get_shape(inputs[0]))
    nout = _numel(get_shape(outputs[0]))
    return 3 * nin + nout                 # approx: (x-μ)^2 + sum + sqrt

def handle_softmax(inputs, outputs):      # aten::softmax
    # Approx: subtract max + exp + sum + log + subtract ≈ 5 ops/element
    n = _numel(get_shape(outputs[0]))     # same shape as input
    return 5 * n                          # approx: sub max, exp, sum, div

def handle_pow(inputs, outputs):          # aten::pow
    return 2 * _numel(get_shape(outputs[0]))

def handle_rsub(inputs, outputs):                # aten::rsub.*
    # y = alpha * rhs - lhs  -> 1 mul + 1 sub per element
    return 2 * _numel(get_shape(outputs[0]))

def handle_linalg_vector_norm(inputs, outputs):  # aten::linalg_vector_norm
    nin  = _numel(get_shape(inputs[0]))
    nout = _numel(get_shape(outputs[0]))         # number of vectors reduced to
    return 2 * nin + nout                        # squares+sum + sqrt per output


def handle_gelu(inputs, outputs):
    return 4 * _numel(get_shape(outputs[0]))

def handle_sdpa(inputs, outputs):
    B,H,N,D = get_shape(inputs[0])
    Nk = get_shape(inputs[1])[2]
    if None in (B,H,N,D,Nk):
        return 0
    fl_qk = 2 * B*H*N*Nk*D
    fl_av = 2 * B*H*N*Nk*D
    fl_sm = B*H*N*Nk
    return fl_qk + fl_av + fl_sm


def flops_breakdown(model, example_input, un_supported_ops, num_train_samples=None,
                    batch_size=1, num_epochs=1):


    """
    Function that counts amount of FLOPs required for all operations inside a model's computation graph.
    For operations that are not supported, some custom FLOP approximations are to be defined before hand, as called from inside this function.

    This measures inference FLOPs for single example input and estimates training FLOPs as 3 * inference_sample flops. It uses other estimations 
    to come up with detailed per_sample, per_epoch, per_trianing FLOPs and returns summary in a dictionary.
    """
    model = model.eval()
    with torch.no_grad(), torch.backends.cuda.sdp_kernel(enable_flash=False,
                                                         enable_mem_efficient=False,
                                                         enable_math=True):
        ana = FlopCountAnalysis(model, (example_input,))

        # Some operations like tanh , sin and log_softmax have been approximated for simplying FLOP count analysis
        element_wise_ops = ("aten::add", "aten::mul", "aten::relu", "aten::sub",
                            "aten::div", "aten::exp","aten::neg","aten::tanh",
                            "aten::sign","aten::abs", "aten::pow", "aten::sqrt", 
                            "aten::log", "aten::sin", "aten::cos"
        )
        ana.set_op_handle("aten::gelu", handle_gelu)
        ana.set_op_handle("aten::scaled_dot_product_attention", handle_sdpa)

        for op in un_supported_ops:
            if op.startswith( element_wise_ops ):
                ana.set_op_handle(op, handle_elemwise)
            elif op.startswith(("aten::rsub",) ):
                ana.set_op_handle(op, handle_rsub)
            elif op.startswith(("aten::sum", "aten::mean")):
                ana.set_op_handle(op, handle_sum if op.startswith("aten::sum") else handle_mean)
            elif op.startswith(("aten::softmax","aten::log_softmax")):
                ana.set_op_handle(op, handle_softmax)
            elif op.startswith(("aten::linalg_vector_norm", "aten::std")):
                ana.set_op_handle(op, handle_linalg_vector_norm if "vector_norm" in op else handle_std)

        fwd_total = ana.total()
        by_mod    = ana.by_module()

        trainable_prefixes = set()
        for name, mod in model.named_modules():
            if any(p.requires_grad for p in mod.parameters(recurse=False)):
                trainable_prefixes.add(name)

        def _under_prefix(n):
            return any(n == p or n.startswith(p + ".") for p in trainable_prefixes)

        fwd_trainable = sum(fl for n, fl in by_mod.items() if _under_prefix(n))

        # per-sample costs
        per_sample_forward = fwd_total
        per_sample_train   = fwd_total + 2 * fwd_trainable

        # scale by batch
        per_step_forward = per_sample_forward * batch_size
        per_step_train   = per_sample_train   * batch_size

        results = {
            "forward_total_per_sample": fwd_total,
            "forward_trainable_per_sample": fwd_trainable,
            "train_per_sample": per_sample_train,
            "forward_per_step": per_step_forward,
            "train_per_step": per_step_train,
            "by_module": by_mod,
        }

        # scale to epoch + full run if dataset size is known
        if num_train_samples is not None:
            steps_per_epoch = (num_train_samples + batch_size - 1) // batch_size
            epoch_train = per_step_train * steps_per_epoch
            full_train  = epoch_train * num_epochs
            results.update({
                "steps_per_epoch": steps_per_epoch,
                "train_per_epoch": epoch_train,
                "train_full": full_train,
                "num_train_samples": num_train_samples,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
            })

        return results
