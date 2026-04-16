function automaticPLYorientation(app,files,target_orientation,enviroment_path)

    for i = 1 : length(files)
        %% PATH
        % To meshlab script
        path_python_scripts = strcat(pwd,filesep,'python',filesep);

        %% Get file
        file_path = strcat(app.input_path,filesep,files(i).Text);

        %% Predict Orientation
        orientation = predictOrientation(app,file,enviroment_path);
    
        %% Rotate to orientation
        command = sprintf('%spython "%srotateCoordinates.py" "%s" "%s" "%s" "%s"',enviroment_path,path_python_scripts,file_path,char(orientation),char(target_orientation),app.output_path);
        system(command)

    end
end
    