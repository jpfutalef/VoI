function [] = paretoPlot(costs, membership)
    % paretoPlot plots "nicely" the pareto front of the costs according to their membership
    % Currently assuming 3 costs

    % Close all figures
    close all;

    % 3D scatter plot with all points
    figure;
    for i = 1:size(costs,1)
        if membership(i)
            scatter3(costs(i,1), costs(i,2), costs(i,3), 'b');
        else
            scatter3(costs(i,1), costs(i,2), costs(i,3), 'r');
        end
        hold on;

        % Create a label for the point as "a{i}"
        point_label = sprintf('a%d', i-1);

        % Add the label to the point and offset a little bit
        text(costs(i,1), costs(i,2), costs(i,3), point_label, 'FontSize', 8, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');

        % Axis labels
        xlabel('L1 (computational load)');
        ylabel('L2 (lack of fit)');
        zlabel('L3 (substitution cost)');
    end

    % three subplots with 2D scatter plots
    figure;

    % L1 vs L2
    subplot(3,1,1);
    for i = 1:size(costs,1)
        if membership(i)
            scatter(costs(i,1), costs(i,2), 'b');
        else
            scatter(costs(i,1), costs(i,2), 'r');
        end
        hold on;

        % Create a label for the point as "a{i}"
        point_label = sprintf('a%d', i-1);

        % Add the label to the point and offset a little bit
        text(costs(i,1), costs(i,2), point_label, 'FontSize', 8, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');

        % Axis labels
        xlabel('L1 (computational load)');
        ylabel('L2 (lack of fit)');
    end

    % L1 vs L3
    subplot(3,1,2);
    for i = 1:size(costs,1)
        if membership(i)
            scatter(costs(i,1), costs(i,3), 'b');
        else
            scatter(costs(i,1), costs(i,3), 'r');
        end
        hold on;

        % Create a label for the point as "a{i}"
        point_label = sprintf('a%d', i-1);

        % Add the label to the point and offset a little bit
        text(costs(i,1), costs(i,3), point_label, 'FontSize', 8, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');

        % Axis labels
        xlabel('L1 (computational load)');
        ylabel('L3 (substitution cost)');
    end

    % L2 vs L3
    subplot(3,1,3);
    for i = 1:size(costs,1)
        if membership(i)
            scatter(costs(i,2), costs(i,3), 'b');
        else
            scatter(costs(i,2), costs(i,3), 'r');
        end
        hold on;

        % Create a label for the point as "a{i}"
        point_label = sprintf('a%d', i-1);

        % Add the label to the point and offset a little bit
        text(costs(i,2), costs(i,3), point_label, 'FontSize', 8, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');

        % Axis labels
        xlabel('L2 (lack of fit)');
        ylabel('L3 (substitution cost)');
    end

end