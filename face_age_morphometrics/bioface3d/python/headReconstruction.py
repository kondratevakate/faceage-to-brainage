import vtk
import sys
import os


if __name__ == "__main__":
   
        nii_file = sys.argv[1]
        output_path = sys.argv[2]
        threshold = sys.argv[3]

        filename_out = os.path.join(output_path,'head_reconstruction.ply')

        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(nii_file)
        reader.Update()

        contour=vtk.vtkFlyingEdges3D()
        contour.SetInputData(reader.GetOutput())
        contour.SetNumberOfContours(1)
        contour.SetValue(0,float(threshold))
        contour.ComputeScalarsOff()
        contour.ComputeNormalsOff()
        contour.ComputeGradientsOff()
        contour.Update()

        # Write in vtk
        triangle = vtk.vtkTriangleFilter()
        triangle.SetInputConnection(contour.GetOutputPort())
        triangle.PassVertsOff()
        triangle.PassLinesOff()

        decimation=vtk.vtkQuadricDecimation()
        decimation.SetInputConnection(triangle.GetOutputPort())

        clean=vtk.vtkCleanPolyData()
        clean.SetInputConnection(triangle.GetOutputPort())

        triangle2 = vtk.vtkTriangleFilter()
        triangle2.SetInputConnection(clean.GetOutputPort())
        triangle2.PassVertsOff()
        triangle2.PassLinesOff()

        #save .vtk polydata
        writer = vtk.vtkPLYWriter()
        writer.SetFileTypeToASCII()
        writer.SetInputConnection(triangle2.GetOutputPort())
        writer.SetFileName(filename_out)
        writer.Write()
