function clip_output=dfe_clipper(input,max_threshold,min_threshold)
if isrow(input)
    max_threshold=max_threshold(:).';
    min_threshold=min_threshold(:).';
else
    max_threshold=max_threshold(:);
    min_threshold=min_threshold(:);
end

clip_output=input;
clip_output(input>max_threshold)=max_threshold(input>max_threshold);
clip_output(input<min_threshold)=min_threshold(input<min_threshold);
