function recolor_plots(ax)
if ~verLessThan('matlab', '8.4.0')
    return
end
colors='brgcmk';
ch=flipud(get(ax, 'children'));

for k=1:length(ch)
    set(ch(k), 'Color', colors(mod(k-1, length(colors))+1));
    set(ch(k), 'LineWidth', 2*floor((k-1)/length(colors))+1);
end
legend (ax, 'off');
warning('off', 'MATLAB:legend:PlotEmpty');
set(legend (ax, 'show'), 'interp', 'none');
