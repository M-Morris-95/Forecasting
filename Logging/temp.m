clear all
root_dir = pwd;
directory = 3;
for directory = 1:3
    if directory == 1
        cd 'ATTENTIONJan-20-10-27'
        figure(1)
    elseif directory == 2
        cd 'ATTENTIONJan-17-15-25'
        figure(2)
    elseif directory == 3
        cd 'GRUJan-20-11-12'
        figure(3)
    else
        cd 'ATTENTIONJan-20-09-35'
        figure(4)
    end
    

    clf
    hold on
    pwd

    d = dir();
    dfolders = d([d(:).isdir]==1);



    shape = size(dfolders)-2;
    num = shape(1);
    base=3;
    for i=base:base-1+num
        loc = string(fullfile(dfolders(i).folder,dfolders(i).name, 'test_predictions.csv'));
        pred = readtable(loc);
    end

    pred = table2array(pred);
    truth = pred(:,3);
    pred(:,3) = [];
    pred(:,1) = [];
    thicc= 5;
    med_thicc = 3;
    plot(truth, 'linewidth',thicc)
    plot(mean(pred, 2), 'linewidth',thicc);
    % plot(mean(pred, 2)-0.5*std(pred, 1, 2), 'linewidth',med_thicc)
    % plot(mean(pred, 2)+0.5*std(pred, 1, 2), 'linewidth',med_thicc)


    legend('Truth', 'Average');
    ylim([0,25]);

    for i =1:num
        plot(pred(:,i))
    end

    [val, idx] = max(pred);
    [tru_val, tru_idx] = max(truth);
    
    R=[];
    for j = 1:num
        for k = j+1:num
            corr = corrcoef(pred(:,i),pred(:,j));

            R(end+1) = corr(2);
        end
    end
    
    x1 = idx;
    y1 = zeros(size(idx));
    x2 = idx;
    y2 = y1 + 25;
    plot_lag = 0;
    if plot_lag == 1
        for i = 1:length(idx)
            plot([x1(i), x2(i)],[y1(i), y2(i)])
        end
    end

    str = ['stddev' , string(mean(std(pred, 1, 2))), 'median= lag',string(median(idx-tru_idx)), 'Corr', string(nanmean(R))];
    title(join(str))

    cd '/Users/michael/Documents/github/Forecasting/Logging/'
end
