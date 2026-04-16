''
%% input example
jstxt=fileread('./aod_otaniemi_16b16.json'); %using 16 strongest paths from RayTrace
aoa_data=jsondecode(jstxt); % ue_count x bts_count x [azimuth,elevatio,amplitude], zeros(3) if no bts
jstxt=fileread('./MLWC/small_otaniemi_matlab_locs.json');
side_data=jsondecode(jstxt); % ue_count x [x,y,bts_count,los_present]
jstxt=fileread('./timing_otaniemi.json'); %using 16 strongest paths from RayTrace
delay_data=jsondecode(jstxt); % ue_count x bts_count x [mean_del_spr,del_spread_var,strong_idx,rss]
logeuc_data=squeeze(cell2mat(struct2cell(readstruct('MLWC/log_euc_9bts_noangle.json')))); %ue_count x ue_count