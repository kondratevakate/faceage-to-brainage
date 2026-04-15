function orientation = predictOrientation(app,file_path,enviroment_path)
    %% PATH
    % To meshlab script
    path_python_scripts = strcat(pwd,filesep,'python',filesep);
    % Load trained model
    model = load(strcat(pwd,filesep,'scripts',filesep,'resources',filesep,'net_resnet50_begin_v2.mat'));
    % Path temp files
    path_temp_files = strcat(strcat(app.output_path,filesep,'_tmp'));

%% Get XY / XZ / YZ projections
    % With python
    command = sprintf('%spython "%sgetProjectionsMOPI.py" "%s" "%s"',enviroment_path,path_python_scripts,file_path,path_temp_files);
    system(command)

    %% STEP4: Generate RGB image
    % Read projections
    path_xy = strcat(path_temp_files,filesep,'_proj_XY.png');
    path_yz = strcat(path_temp_files,filesep,'_proj_YZ.png');
    path_zx = strcat(path_temp_files,filesep,'_proj_ZX.png');

    % Generate BW image
    img_zx = rgb2gray(imread(path_zx));
    img_yz = rgb2gray(imread(path_yz));
    img_xy = rgb2gray(imread(path_xy));

    % Generate image
    img_background = img_xy.*0;

    se = strel('disk',15);
    level = graythresh(img_xy);
    img_zx_bin = imdilate(imbinarize(img_zx,level),se);
    img_yz_bin = imdilate(imbinarize(img_yz,level),se);
    img_xy_bin = imdilate(imbinarize(img_xy,level),se);

    % Select bounding box of projections
    % ZX
    bb_zx  = regionprops(img_zx_bin);
    [~,idx_bb_zx] = sort([bb_zx.Area],'Descend');
    bb_zx_values = bb_zx(idx_bb_zx).BoundingBox;
    bb_zx_values = round(bb_zx_values);
    crop_zx = img_zx(bb_zx_values(2):bb_zx_values(2)+bb_zx_values(4),bb_zx_values(1):bb_zx_values(1)+bb_zx_values(3));
    resize_crop_zx = imresize(crop_zx, [512 512]);
    % YZ
    bb_yz  = regionprops(img_yz_bin);
    [~,idx_bb_yz] = sort([bb_yz.Area],'Descend');
    bb_yz_values = bb_yz(idx_bb_yz).BoundingBox;
    bb_yz_values = round(bb_yz_values);
    crop_yz = img_yz(bb_yz_values(2):bb_yz_values(2)+bb_yz_values(4),bb_yz_values(1):bb_yz_values(1)+bb_yz_values(3));
    resize_crop_yz = imresize(crop_yz, [512 512]);
    % XY
    bb_xy  = regionprops(img_xy_bin);
    [~,idx_bb_xy] = sort([bb_xy.Area],'Descend');
    bb_xy_values = bb_xy(idx_bb_xy).BoundingBox;
    bb_xy_values = round(bb_xy_values);
    crop_xy = img_xy(bb_xy_values(2):bb_xy_values(2)+bb_xy_values(4),bb_xy_values(1):bb_xy_values(1)+bb_xy_values(3));
    resize_crop_xy = imresize(crop_xy, [512 512]);
   
    %Generate RGB divided image
    img_background(1:512,1:512,1) = imrotate(resize_crop_zx,90);
    img_background(1:512,1:512,2) = imrotate(resize_crop_zx,90);
    img_background(1:512,1:512,3) = imrotate(resize_crop_zx,90);
    img_background(1:512,513:1024,1) = resize_crop_yz;
    img_background(1:512,513:1024,2) = resize_crop_yz;
    img_background(1:512,513:1024,3) = resize_crop_yz;
    img_background(513:1024,1:512,1) = imrotate(resize_crop_xy,-90);
    img_background(513:1024,1:512,2) = imrotate(resize_crop_xy,-90);
    img_background(513:1024,1:512,3) = imrotate(resize_crop_xy,-90);
    img_rgb = img_background;
    
    %% PREDICTION
    % Resize image to net
    sz = model.net.Layers(1,1).InputSize;
    img_predict = imresize(img_rgb,sz(1:2));

    %Predict orientation
    % With resnet50
    ori_prediction = classify(model.net,img_predict);
    orientation = string(ori_prediction);
end