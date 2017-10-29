function fig2Pdf(fileName,figWidth,fig)
% fig2Pdf(fileName,figWidth,fig)
% A function to generate pdf file for current figure with the exact size 
if nargin < 3
    fig = gcf;
end 
figSize = get(fig,'Position');
figSize(1:2) = [];

%===Scale the figure to the correct width===
figSize = (figWidth/figSize(1))*figSize; 
set(fig, 'PaperUnits', 'points',...
    'PaperSize',figSize,...
    'PaperPositionMode', 'manual',...
    'PaperPosition',[0 0 figSize]);
print(fig, '-dpdf', '-r300', fileName);

end