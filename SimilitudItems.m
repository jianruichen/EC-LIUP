function [D]=SimilitudItems(data,method)
%INPUT:
%data: rows should be observations while colunms will be variables
%method: metric
%copyright (c) 2010 CONCHA.
%concha.gong@gmail.com

D=zeros(size(data,2),size(data,2));
D1=zeros(size(data,2),size(data,2));
D2=zeros(size(data,2),size(data,2));
switch lower(method)
    case 'cosine'
        for i=1:size(data,2)
            for j=i+1:size(data,2)
                D(i,j)=data(:,i)'*data(:,j)/norm(data(:,i),2)*norm(data(:,j),2);
            end
            if isnan(D(i,j))
                D(i,j)=0;
            end
            D(i,j)=abs(D(i,j));
        end
    case 'correlation'
        for i=1:size(data,2)
            for j=i+1:size(data,2)
                temp=find(data(:,i)~=0 & data(:,j)~=0);
                Rui=data(temp,i);
                Ruj=data(temp,j);
                Ri=mean(data(:,i));
                Rj=mean(data(:,j));
                D(i,j)=(Rui-Ri)'*(Ruj-Rj)/(norm(Rui-Ri)*norm(Ruj-Rj));
                if isnan(D(i,j))
                    D(i,j)=0;
                end
                D(i,j)=abs(D(i,j));
            end
        end
    case 'adjustedcosine'
        for i=1:size(data,2)
            for j=i+1:size(data,2)
                temp=find(data(:,i)~=0 & data(:,j)~=0);
                Rui=data(temp,i);
                Ruj=data(temp,j);
                Ru=mean(data(temp,:)')';
                D(i,j)=(Rui-Ru)'*(Ruj-Ru)/(norm(Rui-Ru)*norm(Ruj-Ru));
                if isnan(D(i,j))
                    D(i,j)=0;
                end
                D(i,j)=abs(D(i,j));
            end
        end
case 'jaccard'
        for i=1:size(data,2)
            for j=i+1:size(data,2)
                temp1=find(data(:,i)~=0 & data(:,j)~=0);
                temp2=find(data(:,i)~=0 | data(:,j)~=0);
                D(i,j)=length(temp1)/length(temp2);
            end
            if isnan(D(i,j))
                D(i,j)=0;
            end
            D(i,j)=abs(D(i,j));
        end
        
    case 'jaccardmsd'
    for i=1:size(data,2)
        for j=i+1:size(data,2)
            temp1=find(data(:,i)>=3 & data(:,j)>=3);
            temp2=find(data(:,i)>=3 | data(:,j)>=3);
            temp3=find(0<data(:,i)<3 & 0<data(:,j)<3);
            temp4=find(0<data(:,i)<3 | 0<data(:,j)<3);
            if ~isempty(temp1)
                %D1(i,j)=(length(temp1)/length(temp2))*(1-(sum(sqrt(data(temp1,i)-data(temp1,j)))/length(temp1)));
                D1(i,j)=(length(temp1)/length(temp2))*(1-(sum(abs(data(temp1,i)-data(temp1,j)))/length(temp1)));
            end
            if ~isempty(temp3)
               %D2(i,j)=(length(temp3)/length(temp4))*(1-(sum(sqrt(data(temp3,i)-data(temp3,j)))/length(temp3)));
               D2(i,j)=(length(temp3)/length(temp4))*(1-(sum(abs(data(temp3,i)-data(temp3,j)))/length(temp3)));
            end
            D(i,j)=D1(i,j);
        end
        if isnan(D(i,j))
            D(i,j)=0;
        end
        D(i,j)=abs(D(i,j));
    end
end
D=D'+D;
            