import SimpleITK as sitk
import os
import sys

if __name__ == "__main__":
    #if len(sys.argv) != 6:
    #    print("Usage: python sliceEnhancement.py path/file.nii outputPath referenceNii")
    #    sys.exit(1)

    nii_file = sys.argv[1]
    output_path = sys.argv[2]
    reference_nii = sys.argv[3]

    ## HISTOGRAM MATCHING
    raw_img = sitk.ReadImage(nii_file, sitk.sitkFloat32)
    raw_img_sitk = sitk.DICOMOrient(raw_img,'LSA')

    ### Histogram Matching - HM ###
    template_img_path = reference_nii
    template_img_sitk = sitk.ReadImage(template_img_path, sitk.sitkFloat32)
    template_img_sitk = sitk.DICOMOrient(template_img_sitk,'LSA')
    hm = sitk.HistogramMatching(raw_img_sitk, template_img_sitk)

    file = os.path.basename(nii_file)
    hm_resampled = sitk.Resample(hm, raw_img_sitk , sitk.Transform(), sitk.sitkLinear, 0.0, hm.GetPixelID())
    sitk.WriteImage(hm_resampled, os.path.join(output_path,'mri_enhanced.nii'))