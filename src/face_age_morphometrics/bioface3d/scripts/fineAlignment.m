function fineAlignment(app,file_ply,virtual_env)
    %% PATHs
    path_mvcnn_scripts = strcat(pwd,filesep,'mvcnn',filesep);

    %% READ REFERENCE MATCHPOINTS
    points1 = readmatrix(strcat(pwd,filesep,'scripts',filesep,'resources',filesep,'reference_controlPoints.txt'));
    matchedPoints1 = points1([21 29 46 47 53],:);

    %% PROCESS
    
    % Read input mesh
    [vertices_mov,faces_mov] = plyRead(file_ply,1);
    ptCloudTformed = pointCloud(vertices_mov);
    
    % Calculate matchepoints2 with MVCNN
    model_path = strcat(path_mvcnn_scripts,filesep,'__configs',filesep,'DTU3D',filesep,'DTU3D_url_depth.json');
    command = sprintf('%spython "%spredict.py" -c "%s" -n "%s" -pn 1 -pt 1 -mr 100',virtual_env,path_mvcnn_scripts,model_path,file_ply);
    system(command)
    
    % Check if prediction has been done
    [folder_ply, name_ply, ~] = fileparts(file_ply);
    if isfile(strcat(folder_ply,filesep,name_ply,'_landmarks.txt'))
        % Read matchedpoints2
        points2 = readmatrix(strcat(folder_ply,filesep,name_ply,'_landmarks.txt'));
        pc_points2 = pointCloud(points2);
        matchedPoints2 = points2([21 29 46 47 53],:);

        %% ALIGNMENT
        % Estimate transform amtrix
        [tformEst,~] = estimateGeometricTransform3D(matchedPoints1, ...
        matchedPoints2,'similarity','MaxDistance',20); 

        % Apply trnasformation
        ptCloudOut = pctransform(ptCloudTformed,invert(tformEst));
        pc_points2_out = pctransform(pc_points2,invert(tformEst));

        % Save data
        dir_out = strcat(app.output_path,filesep,'_head_alignment');
        if exist(dir_out, 'dir') ~= 7
            mkdir(dir_out);
        end
        plyWrite(ptCloudOut.Location(:,:),faces_mov,strcat(app.output_path,filesep,'_head_alignment',filesep,name_ply,'.ply'));
        %writematrix(pc_points2_out.Location(:,:),[path_temp_files '\_05_oriented_landmarks_aligned.txt'],'Delimiter',' ')

        % Remove landmarking files
        try
            delete(strcat(folder_ply,filesep,name_ply,'_landmarks.txt'));
            delete(strcat(folder_ply,filesep,name_ply,'_landmarks.landmarkAscii'));
            delete(strcat(folder_ply,filesep,name_ply,'_ransac.txt'));
        catch
             disp('Error deleting file(s)');
        end
    else
        %If file not exists, final alignment is cancelled
        disp('Final alignment will not be taken into account')
        %copyfile([path_temp_files '\_05_oriented.ply'],[path_temp_files '\_06_aligned.ply']);
    end
   
end