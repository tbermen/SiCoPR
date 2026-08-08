function hard_cap = compute_hard_cap(use_hard_cap, mul, LSV, min_radius)

if use_hard_cap
    hard_cap = max(min_radius, round(mul * LSV));
else
    hard_cap = NaN;
end
