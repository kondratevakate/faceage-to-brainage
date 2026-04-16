function faceExtraction(app,file_ply,virtual_env)
    %% PATHs
    path_mvcnn_scripts = strcat(pwd,filesep,'mvcnn',filesep);
    path_python_scripts = strcat(pwd,filesep,'python',filesep);

    %% PROCESS
      
    % Calculate matchepoints2 with MVCNN
    model_path = strcat(path_mvcnn_scripts,filesep,'__configs',filesep,'DTU3D',filesep,'DTU3D_url_depth.json');
    command = sprintf('%spython "%spredict.py" -c "%s" -n "%s" -pn 1 -pt 1 -mr 100',virtual_env,path_mvcnn_scripts,model_path,file_ply);
    system(command)
    
    % Check if prediction has been done
    [folder_ply, name_ply, ~] = fileparts(file_ply);
    if isfile(strcat(folder_ply,filesep,name_ply,'_landmarks.txt'))
        controlPoints = readmatrix(strcat(folder_ply,filesep,name_ply,'_landmarks.txt'));
        %Set threshold
        cp_chin = -controlPoints(68,2);
        threshold_y = -cp_chin-10;
        threshold_z = 40;
        %Delete files
        try
            delete(strcat(folder_ply,filesep,name_ply,'_landmarks.txt'));
            delete(strcat(folder_ply,filesep,name_ply,'_landmarks.landmarkAscii'));
            delete(strcat(folder_ply,filesep,name_ply,'_ransac.txt'));
        catch
             disp('Error deleting file(s)');
        end
    else
         pc = pcread(file_ply);
         limit_y = pc.YLimits(2);
         threshold_y = -limit_y;
         threshold_z = 40;
    end

    dir_out = strcat(app.output_path,filesep,'_face_extraction');
    if exist(dir_out, 'dir') ~= 7
        mkdir(dir_out);
    end
    path_out = strcat(app.output_path,filesep,'_face_extraction',filesep,name_ply,'.ply');
    path_temp_files = strcat(app.output_path,filesep,'_tmp');
    command = sprintf('%spython "%sfaceExtraction.py" "%s" "%f" "%f" "%s"',virtual_env,path_python_scripts,file_ply,threshold_z,threshold_y,path_out);
    system(command)

    % Plot results on GUI
    % Generate projections
    % With python
    command = sprintf('%spython "%sgetProjectionsMOPI.py" "%s" "%s"',virtual_env,path_python_scripts,path_out, path_temp_files);
    system(command)

    % Read projections
    path_xy = strcat(path_temp_files,filesep,'_proj_XY.png');
    path_yz = strcat(path_temp_files,filesep,'_proj_YZ.png');

    % Update GUI
    app.Image.ImageSource = imread(path_xy);
    app.Image2.ImageSource = imrotate(imread(path_yz),90);
    app.ThresholdYSpinner.Value = threshold_y;
    app.ThresholdZSpinner.Value = threshold_z;
       
end