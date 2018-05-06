for i = 1:50
    string = strcat('Patient_1_interictal_segment_00', num2str(i, '%02d'));
    %load(strcat(string, '.mat'));
    %csvwrite(strcat(string, '.csv'), string);
    disp(string);
end