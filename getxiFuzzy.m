function [Fweight] = getxiFuzzy(xi,option1)
%GETXIFUZZY 此处显示有关此函数的摘要
%   此处显示详细说明
delta=1e-3;

if option1 == 1
    Fweight1=1./(1+xi);
elseif option1 == 2
    maxxi=max(xi);
    Fweight1=1-xi./(maxxi+delta);
end
Fweight=Fweight1';
end

