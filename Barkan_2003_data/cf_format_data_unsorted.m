clc
clear

% nsub = 100
% 33 choices per subj = 16 gambles played twice plus ne gamble at start
% data is a nsub*33 X 12: g l trialpay stagepay
%    wptt wptn wpnt wpnn lptt lptn lpnt lpnn
% avg log like ranges from -104 to -37

load BarkanData  % contains a file called: data
data_reshaped = reshape(data,100,33,12);
n_subj = size(data_reshaped, 1)
n_gambles = size(data_reshaped, 2)
n_cols = size(data_reshaped,3) + 2

data(1:33,:) == squeeze(data_reshaped(1,:,:))
formatted_data = []

for s = 1:n_subj
    temp = [repmat(s, n_gambles, 1) transpose(1:n_gambles) squeeze(data_reshaped(s,:,:))];
    formatted_data = [formatted_data; temp];
end

csvwrite("Barkan_2003_data_unsorted.csv", formatted_data)
