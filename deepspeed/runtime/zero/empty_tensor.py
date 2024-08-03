import torch
from torch.utils._pytree import tree_map
import contextlib


@contextlib.contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard


tensor_map = {}


class EmptyTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
            dtype=elem.dtype,
            layout=elem.layout,
            requires_grad=elem.requires_grad,
            device=elem.device,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def inflate(t):
            if isinstance(t, cls):
                with no_dispatch():
                    return torch.ones_like(t)
            else:
                return t

        args = tree_map(inflate, args)
        kwargs = tree_map(inflate, kwargs)

        # https://github.com/pytorch/pytorch/issues/77265
        if func is torch.ops.aten.detach.default:
            # Special handling for detach
            result = args[0].detach()
            return EmptyTensor(result)

        return super().__torch_dispatch__(
                func, types, args, kwargs
        )
