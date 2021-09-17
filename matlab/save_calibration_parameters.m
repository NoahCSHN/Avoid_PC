function save_calibration_parameters(file_path)
%% 
left_intr_matrix = stereoParams.CameraParameters1.IntrinsicMatrix
right_intr_matrix = stereoParams.CameraParameters2.IntrinsicMatrix
left_rad_dis = stereoParams.CameraParameters1.RadialDistortion
right_rad_dis = stereoParams.CameraParameters2.RadialDistortion
left_tan_dis = stereoParams.CameraParameters1.TangentialDistortion
right_tan_dis = stereoParams.CameraParameters2.TangentialDistortion
right_rot_matrix = stereoParams.RotationOfCamera2
right_tra_matrix = stereoParams.TranslationOfCamera2

%%
file_name = fullfile(file_path,'left_intr_matrix.mat');
save(file_name,'left_intr_matrix');
file_name = fullfile(file_path,'right_intr_matrix.mat');
save(file_name,'right_intr_matrix');
file_name = fullfile(file_path,'left_rad_dis.mat');
save(file_name,'left_rad_dis');
file_name = fullfile(file_path,'right_rad_dis.mat');
save(file_name,'right_rad_dis');
file_name = fullfile(file_path,'left_tan_dis.mat');
save(file_name,'left_tan_dis');
file_name = fullfile(file_path,'right_tan_dis.mat');
save(file_name,'right_tan_dis');
file_name = fullfile(file_path,'right_rot_matrix.mat');
save(file_name,'right_rot_matrix');
file_name = fullfile(file_path,'right_tra_matrix.mat');
save(file_name,'right_tra_matrix');
end