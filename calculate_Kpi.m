function [Acc] = calculate_Kpi( preY,testLabel,imbanlance)
% 得到精度
% 评价多分类问题时，通常把多分类问题分解成多个2分类问题。
% 即n分类，分解为n个2分类，每次以其中一个类为正类，其余类统一为负类
% 计算之前提到的各种2分类指标，最后再平均计算多分类评价指标，有三种平均方式
% Macro 宏平均；Micro 微平均 将所有类直接放到一起来计算；weight 加权平均 各个类别要乘以该类在总样本中的占比来求和
% https://blog.csdn.net/sdu_hao/article/details/103533115

flag=1;
% flag=1 Macro 宏平均 分别计算第i类然后进行平均
% flag=2 Micro 多分类的accuracy，recall和precision会相同
% flag=3 Weighted 解决Macro中没有考虑样本不均衡的原因

if imbanlance ==1
    Acc = Gmean(preY,testLabel);
elseif imbanlance == 0
    accuacy = length(find(preY==testLabel))/length(testLabel);
    Gmean1 = Gmean(preY,testLabel);
    Acc=[accuacy,Gmean1];
elseif imbanlance == -1
    accuacy = length(find(preY==testLabel))/length(testLabel);
    Gmean1 = Gmean(preY,testLabel);
    C = confusionmat(testLabel,preY);
    [Result_special,ReferenceResult] = multiclass_metrics_special(C);
    %Acc,G-mean,Recall,Precision,Specificity,
    %FalsePositiveRate,F1_score,MatthewsCorrelationCoefficient,Kappa
    Acc=[accuacy,Gmean1,Result_special.Recall,Result_special.Precision,Result_special.Specificity,...
        Result_special.FalsePositiveRate,Result_special.F1_score,Result_special.MatthewsCorrelationCoefficient,Result_special.Kappa];% Specificity不适合多分类问题
end