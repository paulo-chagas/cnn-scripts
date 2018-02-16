base = 'validacao2';
 
%n_amostras = 4860;
labels = {'arc', 'area', 'bar', 'coord_parallel', 'line', 'matrix', 'pie', 'scatter', 'sunburst', 'treemap'};
%images = cell(1,n_amostras);
%classes = cell(1,n_amostras);
%classes_index = zeros(1, n_amostras);
%hogFeatures = cell(1, n_amostras);

fid = fopen([base '-28x28.arff'],'w');

fprintf(fid, '@RELATION %s \n\n', base);
for i=1:1296
    fprintf(fid, '@ATTRIBUTE att%s NUMERIC\n', num2str(i));
end

list_classes = strjoin(cellstr(labels),',');
fprintf(fid, '@ATTRIBUTE class {%s} \n\n@data\n', list_classes);

for l=1:length(labels)
    label = labels{l};
    disp(['Reading ', label, ' charts'])
    subdir = ['.\',base,'\',label, '\*.png'];
    imagefiles = dir(subdir);
    nfiles = length(imagefiles);    % Number of files found
    disp([num2str(nfiles), ' ', label, ' charts  found'])
    
    for ii=1:nfiles
       currentfilename = imagefiles(ii).name;
       %disp(['Nome: ', num2str(currentfilename)])
       currentimage = imread(['.\',base,'\',label, '\',currentfilename]);
       currentimage = imresize(currentimage, [200 200]);
       if size(currentimage,3)==3
           img = rgb2gray(currentimage);
       else
           img = currentimage;
       end
       %img = imbinarize(currentimage);
       %threshold = graythresh(currentimage);
       %img = im2bw(currentimage,threshold); 
       
       features = extractHOGFeatures(img, 'CellSize',[28 28]);
       
       s = strjoin(arrayfun(@(x) num2str(x),features,'UniformOutput',false),',');
       s = strcat(s,[',' label]);
       fprintf( fid, '%s\n',s);
       
       %hogFeatures{ii} = features;
       %images{ii} = img;
       %classes{ii} = label;
       %classes_index(ii) = l;
    end
end
disp('Done');
fclose(fid);