function [slice] = getNSlice(file, N, orientation)
    % Read the NIfTI file
    niftiData = niftiread(file);
    
    % Extract the central slices along each dimension
    switch(orientation)
        case 'xSlice'
            slice = squeeze(niftiData(:, :, N, 1));
        case 'ySlice'
            slice = squeeze(niftiData(:, N, :, 1));
        case 'zSlice'
            slice = squeeze(niftiData(N, :, :, 1));
    end
end