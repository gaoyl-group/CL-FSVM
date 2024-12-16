function [Acc,svm] = testFSVM( trainData,trainLabel,testData,testLabel,kertype,C,imbanlance,alphaCoeff,paraLambda,m_1)
class = unique(trainLabel,'stable');
nuclass = length(class);
MaxCount=10;
epsilon=1e-4;
LossTable=zeros(1,MaxCount);

if(nuclass == 2)
    trainLabel = mapminmax(trainLabel,-1,1);
    index1 = find(trainLabel == 1); index2 = find(trainLabel == -1);
    trainData1 = trainData(:,index1);trainLabel1 = trainLabel(:,index1);
    trainData2 = trainData(:,index2);trainLabel2 = trainLabel(:,index2);
    trainData = [trainData1,trainData2];trainLabel = [trainLabel1,trainLabel2];

    testLabel = mapminmax(testLabel,-1,1);

    if(strcmp(kertype,'linear'))
        SVMModel = fitcsvm(trainData',trainLabel');
        svm.w=SVMModel.Beta';
        svm.b=SVMModel.Bias;
    else
        SVMModel = fitcsvm(trainData',trainLabel','KernelFunction','gaussian');
        svm.a=SVMModel.Alpha;
        svm.b=SVMModel.Bias;
        svm.Xsv=SVMModel.SupportVectors';
        svm.Ysv=SVMModel.SupportVectorLabels';
    end

    knnType=-1;
    G = zeros(length(trainLabel),1);
    Fweight = ones(length(trainLabel),1);
    [~,xi] = getFunctionLoss(svm,trainData,kertype,trainLabel,G,paraLambda,Fweight,C);
    [Si] = cal_S( trainData',trainLabel',alphaCoeff,knnType,nuclass,nuclass);
    count=0;
    while(count<MaxCount)
        count=count+1;
        Fweight1 = getxiFuzzy(xi,2);
        [G,Fweight] = RegularizerLearning(trainData',trainLabel',svm,kertype,Si,paraLambda,m_1,knnType,Fweight1,paraLambda);

        [~,svm] = trainFSVM( trainData1,trainLabel1,trainData2,trainLabel2,Fweight,kertype,C,G,paraLambda);
        a=svm.aAll;
        [loss1,xi] = getFunctionLoss(svm,trainData,kertype,trainLabel,G,paraLambda,Fweight,C);
        LossTable(count)=loss1;

        if(count>1 && abs(LossTable(count)-LossTable(count-1))/max(1,LossTable(1))<=epsilon)
            %i=i
            LossTable((count+1):MaxCount) = LossTable(count);
            break;
        elseif(count>1 && LossTable(count)/max(1,LossTable(1))>LossTable(count-1)/max(1,LossTable(1))+epsilon)
            a = aa;
            G = GG;
            xi = xxi;
            Fweight = getxiFuzzy(xi,2);
            svm = ssvm;
            LossTable(count) = 0;
            count = count - 1;
            %Armijo rule
            df = dfda(a,G,trainData,trainLabel,kertype,paraLambda);
            stepsize = 1e-3;
            while(loss1/max(1,LossTable(1))>LossTable(count)/max(1,LossTable(1)) - epsilon && stepsize > 1e-8)
                stepsize = stepsize*0.01;
                a = aa + stepsize*df;
                %update svm to get di
                [svm,~] = calculate_rho(a,trainData',trainLabel',C,kertype,paraLambda);
                [loss1,xi] = getFunctionLoss(svm,trainData,kertype,trainLabel,G,paraLambda,Fweight,C);
            end
            if(stepsize < 1e-7)
                svm = ssvm;
                LossTable((count+1):MaxCount) = LossTable(count);
                break;
            end
            aa = a;
            GG = G;
            xxi = xi;
            ssvm = svm;
            count = count + 1;
            LossTable(count) = loss1;
        else
            aa = a;
            GG = G;
            xxi = xi;
            ssvm = svm;
        end
    end
    SVs = svm.svnum;
    result=svmTest_multiclass(svm,testData,kertype,paraLambda);
    testTime = toc;
    preY = sign(result.score);
    Acc = calculate_Kpi( preY,testLabel,imbanlance);
elseif(nuclass > 2) %max(f(x))
    trainData0=trainData;
    trainLabel0=trainLabel;
    for ii = 1:nuclass
        iclass = class(ii);
        index1 = find(trainLabel0==iclass);
        %one v all
        index2 = find(trainLabel0~=iclass);
        trainLabel1 = trainLabel0(index1) - trainLabel0(index1) + 1;%one为1
        trainLabel2 = trainLabel0(index2) - trainLabel0(index2) - 1;%All为-1
        trainData1 = trainData0(:,index1);
        trainData2 = trainData0(:,index2);
        trainData = [trainData1,trainData2];trainLabel = [trainLabel1,trainLabel2];

        if(strcmp(kertype,'linear'))
            SVMModel = fitcsvm(trainData',trainLabel');
            svm.w=SVMModel.Beta';
            svm.b=SVMModel.Bias;
        else
            SVMModel = fitcsvm(trainData',trainLabel','KernelFunction','gaussian');
            svm.a=SVMModel.Alpha;
            svm.b=SVMModel.Bias;
            svm.Xsv=SVMModel.SupportVectors';
            svm.Ysv=SVMModel.SupportVectorLabels';
        end


        knnType=-1;
        G = zeros(length(trainLabel),1);
        Fweight = ones(length(trainLabel),1);
        [~,xi] = getFunctionLoss(svm,trainData,kertype,trainLabel,G,paraLambda,Fweight,C);
        [Si] = cal_S( trainData',trainLabel',alphaCoeff,knnType,nuclass,ii);
        count=0;
        while(count<MaxCount)
            count=count+1;
            Fweight1 = getxiFuzzy(xi,2);
            [G,Fweight] = RegularizerLearning(trainData',trainLabel',svm,kertype,Si,paraLambda,m_1,knnType,Fweight1,paraLambda);
            [~,svm] = trainFSVM( trainData1,trainLabel1,trainData2,trainLabel2,Fweight,kertype,C,G,paraLambda);
            a=svm.aAll;
            [loss1,xi] = getFunctionLoss(svm,trainData,kertype,trainLabel,G,paraLambda,Fweight,C);
            LossTable(count)=loss1;

            if(count>1 && abs(LossTable(count)-LossTable(count-1))/max(1,LossTable(1))<=epsilon)
                %i=i
                LossTable((count+1):MaxCount) = LossTable(count);
                break;
            elseif(count>1 && LossTable(count)/max(1,LossTable(1))>LossTable(count-1)/max(1,LossTable(1))+epsilon)
                a = aa;
                G = GG;
                svm = ssvm;
                xi = xxi;
                Fweight = getxiFuzzy(xi,2);
                LossTable(count) = 0;
                count = count - 1;
                %Armijo rule
                df = dfda(a,G,trainData,trainLabel,kertype,paraLambda);
                stepsize = 1e-3;
                while(loss1/max(1,LossTable(1))>LossTable(count)/max(1,LossTable(1)) - epsilon && stepsize > 1e-8)
                    stepsize = stepsize*0.01;
                    a = aa + stepsize*df;
                    %update svm to get di
                    [svm,~] = calculate_rho(a,trainData',trainLabel',C,kertype,paraLambda);
                    [loss1,xi] = getFunctionLoss(svm,trainData,kertype,trainLabel,G,paraLambda,Fweight,C);
                end
                if(stepsize < 1e-7)
                    svm = ssvm;
                    LossTable((count+1):MaxCount) = LossTable(count);
                    break;
                end
                aa = a;
                GG = G;
                ssvm = svm;
                xxi = xi;
                count = count + 1;
                LossTable(count) = loss1;
            else
                aa = a;
                GG = G;
                xxi = xi;
                ssvm = svm;
            end
        end
        SVs = svm.svnum;
        result=svmTest_multiclass(svm,testData,kertype,paraLambda);
        testTime = toc;
        testYList(ii,:) = result.score;
    end
    [maxLabel,maxIndex] = max(testYList);
    preY = class(maxIndex);
    Acc = calculate_Kpi( preY,testLabel,imbanlance);

end
end
