def OptFom_Set_Best_Itick(THIS, BEST):
    """Update BEST struct FOM tracking for positive/negative itick (MATLAB lines 3785-3803)."""
    FOM = THIS.FOM
    itick = THIS.itick
    if FOM > BEST.itick_FOM:
        BEST.itick_FOM = FOM
        BEST.itick_in_cluster = itick
    if itick >= 0 and FOM > BEST.positive_itick_FOM:
        BEST.positive_itick_FOM = FOM
        BEST.positive_itick_in_loop = itick
    if itick <= 0 and FOM > BEST.negative_itick_FOM:
        BEST.negative_itick_FOM = FOM
        BEST.negative_itick_in_loop = itick
    return BEST
