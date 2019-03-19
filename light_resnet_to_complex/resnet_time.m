cmp = importdata('re10_heav32_2000_record.csv');
cmp = cmp.data;
tran = importdata('light16_to_heav32_transfer_2000_record.csv');
tran = tran.data;


hold on
scatter(1+0.02*rand(10,1), cmp);
scatter(2+0.02*rand(10,1), tran);

boxplot([cmp, tran],'Labels',{'R-ResNet','T-ResNet'});
box off
ylabel('Training time(s)')

%%
 r_t = mean(cmp); t_t = mean(tran);
bar_data = [r_t,t_t];


 b = bar( bar_data,'FaceColor','flat');
 b.CData(2,:) = [0 0.8 0.9];
title('R-ResNet vs T-ResNet')
ylabel('Training time(s)')
xticklabels({'R-ResNet','T-ResNet'})
text(1-0.3,r_t+20, [num2str(r_t), 's'])
text(2-0.3,t_t+20, [num2str(t_t), 's'])
%     text(2-0.3,t_t+b_t+30, num2str(t_t+b_t))
