% This file contains quick and dirty behavioural explorations of the dataset. 
% LOAD DATA

partial_paths = ["20160929" "20160930" "20161005" "20161017"; "20160112" "20160113" "20160121" "20160122"]; 
names = ["V", "W"]; 
for n = 1:2
  name = names(n);
  behav = []
  for j = 1:4
    path = strcat("SPKcounts_", name, partial_paths(n,j), "cue_MW_250X250ms.mat");
    overall = load(path);
    behav1 = overall.Y;
    RT = overall.ReactTargetRewardTimes;
    % Concatenate the behavioral data with RT
    behav1 = [behav1 RT];
      
    behav = [behav; behav1];
  end

  behav = behav(behav(:,13)==1,:);
  % First plot: Panel of 4; 1) RT histogram across. 2) RT vs. Past Reward. 3) RT vs. Time (before the 40th trial) 4) RT vs. Time (from the 40th trial). 
  % 1) RT histogram across
  figure(n); 
  % overall figure title
  title(strcat("Behavioral Exploration of ", name, " session ", string(j)));
  subplot(3,2,1);
  histogram(behav(:,14), 50);
  title('RT histogram across all trials');
  xlabel('RT (ms)');
  ylabel('Frequency');

  % 2) RT vs. Past Reward
  subplot(3,2,2);
  % set alpha to be 0.1
  % scatter(behav(1:(length(behav) - 1),3), behav(2:length(behav),14), "Color", [0, 0.4470, 0.7410, 0.1]);
  changed = [0]; % Default the first trial to be not changed
  for i = 2:length(behav)
      dimension = behav(i,10);
      if (behav(i,dimension) == 0 && behav(i-1,dimension) == 1) || (behav(i,dimension) == 1 && behav(i-1,dimension) == 0)
          changed = [changed; 1];
      else
          changed = [changed; 0];
      end
  end
  behav(:, 17) = changed; 
  behav(:, 18) = [0; behav(1:length(behav) - 1,3)];

  changed_rewarded = behav(behav(:, 17) == 1 & behav(:, 18) == 1, :);
  changed_not_rewarded = behav(behav(:, 17) == 1 & behav(:, 18) == 0, :);
  unchanged_rewarded = behav(behav(:, 17) == 0 & behav(:, 18) == 1, :);
  unchanged_not_rewarded = behav(behav(:, 17) == 0 & behav(:, 18) == 0, :);

  group = [repmat({'Not Rewarded - Unchanged'}, size(unchanged_not_rewarded, 1), 1); 
           repmat({'Not Rewarded - Changed'}, size(changed_not_rewarded, 1), 1); 
           repmat({'Rewarded - Unchanged'}, size(unchanged_rewarded, 1), 1); 
           repmat({'Rewarded - Changed'}, size(changed_rewarded, 1), 1)];

  rt = [unchanged_not_rewarded(:,14); changed_not_rewarded(:,14); unchanged_rewarded(:,14); changed_rewarded(:,14)];

  boxplot(rt, group);
  title('RT vs. Past Reward and Choice Change');
  xlabel('Condition');
  ylabel('RT (ms)');

  % 3) RT vs. Time (before the 40th trial)
  % Color by the past reward
  subplot(3,2,3);
  rewarded = behav(boolean([1 transpose(behav(1:(length(behav)-1),3) == 1)]), :);
  hold off
  scatter(rewarded(rewarded(:,6)<40,6), rewarded(rewarded(:,6)<40,14), "Color", [0, 0.4470, 0.7410, 0.1]);
  hold on
  unrewarded = behav(boolean([1 transpose(behav(1:(length(behav)-1),3) == 0)]), :);
  scatter(unrewarded(unrewarded(:,6)<40,6), unrewarded(unrewarded(:,6)<40,14), "Color", [0.8500, 0.3250, 0.0980, 0.1]);
  title('RT vs. Time (before the 40th trial)');
  legend('Rewarded', 'Unrewarded');
  xlabel('Time (trial)');
  % set y limits
  ylim([0, max(behav(:,14))]);
  ylabel('RT (ms)');

  % 4) RT vs. Time (from the 40th trial)
  % Color by the past reward
  subplot(3,2,4);
  hold off
  scatter(rewarded(rewarded(:,6)>=40,6), rewarded(rewarded(:,6)>=40,14));
  hold on
  scatter(unrewarded(unrewarded(:,6)>=40,6), unrewarded(unrewarded(:,6)>=40,14));
  legend('Rewarded', 'Unrewarded');
  title('RT vs. Time (from the 40th trial)');
  xlabel('Time (trial)');
  % set y limits
  ylim([0, max(behav(:,14))]);
  ylabel('RT (ms)');

  % 5) choice change following no reward
  subplot(3,2,5);
  change_time = [];
  stick_time = [];
  no_reward = behav(behav(:,3) == 0, :);
  for i = 1:length(no_reward) - 1
      dimension = no_reward(i,10);
      if (no_reward(i,dimension) == 0 && no_reward(i+1,dimension) == 1) || (no_reward(i,dimension) == 1 && no_reward(i+1,dimension) == 0)
          change_time = [change_time; no_reward(i,6)];
      else
          stick_time = [stick_time; no_reward(i,6)];
      end
  end

  hold off
  histogram(change_time, FaceAlpha = 0.5);
  hold on
  histogram(stick_time, FaceAlpha = 0.5);
  title('Choice change following no reward');
  xlabel('Time (trial)');
  ylabel('Frequency');
  legend('Change', 'Stick');

  % 6) choice change following reward
  subplot(3,2,6);
  change_time = [];
  stick_time = [];
  no_reward = behav(behav(:,3) == 1, :);
  for i = 1:length(no_reward) - 1
      dimension = no_reward(i,10);
      if (no_reward(i,dimension) == 0 && no_reward(i+1,dimension) == 1) || (no_reward(i,dimension) == 1 && no_reward(i+1,dimension) == 0)
          change_time = [change_time; no_reward(i,6)];
      else
          stick_time = [stick_time; no_reward(i,6)];
      end
  end
  hold off
  histogram(change_time, FaceAlpha = 0.5);
  hold on
  histogram(stick_time, FaceAlpha = 0.5);
  title('Choice change following reward');
  xlabel('Time (trial)');
  ylabel('Frequency');
  legend('Change', 'Stick');
end


