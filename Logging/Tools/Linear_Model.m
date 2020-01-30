clear all
root_dir = pwd;
root_dir = root_dir(1:end-14);

google_train = readtable(string(fullfile(root_dir, 'google_train.csv')));
google_test = readtable(string(fullfile(root_dir, 'google_test.csv')));
y_train = readtable(string(fullfile(root_dir, 'y_train.csv')));
y_test = readtable(string(fullfile(root_dir, 'y_test.csv')));

test_mean = google_test.mean;
test_max = google_test.max;
test_min = google_test.min;
test_day = google_test.Unnamed_0;
google_test.Unnamed_0 = [];
google_test.min = [];
google_test.mean = [];
google_test.max = [];
google_test.Var1 = [];

train_mean = google_train.mean;
train_max = google_train.max;
train_min = google_train.min;
train_day = google_train.Unnamed_0;
google_train.Unnamed_0 = [];
google_train.min = [];
google_train.mean = [];
google_train.max = [];
google_train.Var1 = [];

y_train = y_train.Var3;
y_train(1) = [];
y_test = y_test.Var3;
y_test(1) = [];

%FIX Y_TEST
temp = zeros(length(y_test),1);
for i = 1:length(y_test)
    temp(i) = str2num(string(y_test(i)));
end
y_test = zeros(length(temp)-21,21);
for i = 1:length(y_test)
    y_test(i,:) = temp(i:i+20);
end

%FIX X_TEST
temp =  table2array(google_test);

google_test = zeros(length(temp)-27,28, 176);
for i = 1:length(google_test)
    google_test(i,:,:) = temp(i:i+27,:);
end

%FIX Y_TRAIN
temp = zeros(length(y_train),1);
for i = 1:length(y_train)
    temp(i) = str2num(string(y_train(i)));
end
y_train = zeros(length(temp)-21,21);
for i = 1:length(y_train)
    y_train(i,:) = temp(i:i+20);
end

%FIX X_TEST
temp =  table2array(google_train);
google_train = zeros(length(temp)-27,28, 176);
for i = 1:length(google_train)
    google_train(i,:,:) = temp(i:i+27,:);
end







