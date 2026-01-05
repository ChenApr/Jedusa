# medusa_jittor_all/model/medusa_mask.py
import jittor as jt

def pad_medusa_mask(mask_qq: jt.Var, tree_step: int) -> jt.Var:
    """
    mask_qq: [q,q] or [1,1,q,q] (float/bool ok)
    return:  [1,1,tree_step,tree_step] float32 (0/1)
    """
    if mask_qq is None:
        return None

    if len(mask_qq.shape) == 4:
        m = mask_qq
    else:
        m = mask_qq.unsqueeze(0).unsqueeze(0)

    q = int(m.shape[-1])
    T = int(tree_step)

    # to float32 0/1
    m = (m > 0).cast(jt.float32)

    if q == T:
        return m

    out = jt.zeros((1, 1, T, T), dtype=jt.float32)
    qq = min(q, T)
    out[:, :, :qq, :qq] = m[:, :, :qq, :qq]
    return out