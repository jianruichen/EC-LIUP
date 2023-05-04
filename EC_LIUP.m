function [MAE,RMSE]=EC_LIUP(mycase,neibor_num,method)
%% parameters
iter=100;
K1=0.1;
K2=0.1;
sigma=0.2;
Ucluster_number=3;
%% load data
if mycase==1
   load new\u1base
elseif mycase==2
   load new\u2base
elseif mycase==3
   load new\u3base
elseif mycase==4
   load new\u4base
elseif mycase==5
   load new\u5base
end
load S1;
load S2;
m=size(u1,1);
number_user=max(u1(:,1));
number_movies=max(u1(:,2));
N=number_user+number_movies;
score_matrix=zeros(number_user,number_movies);
for i=1:m
    score_matrix(u1(i,1),u1(i,2))=u1(i,3);
end
user_degree=zeros(1,number_user);
for i=1:number_user
    user_degree(i)=length(find(score_matrix(i,:)>=3));
end
mean_user=mean(user_degree);
item_degree=zeros(1,number_movies);
for j=1:number_movies
    item_degree(j)=length(find(score_matrix(:,j)>=3));
end
mean_item=mean(item_degree);
degree = [user_degree,item_degree];
adj_matrix=[S1,score_matrix;score_matrix',S2];
%% evolutionary clustering
y = zeros(iter,N);
y(1,:)=rand(1,N).*pi/2;
y(2,:)=rand(1,N).*pi/2;
for e = 2:iter
   for i = 1:N
        a = 0;b = 0;
        a1 = 0;b1 = 0;
        for j = 1:number_user
            if adj_matrix(i,j) >= 3 && degree(j)>=mean_user
                a = a + degree(j)/number_user*sin(adj_matrix(i,j)*(y(e,j) - y(e,i)));
                a1 = a1 + degree(j)/number_user*sin(adj_matrix(i,j)*(y(e-1,j) - y(e,i)));  
            end
        end
        for j = number_user+1:N
            if adj_matrix(i,j) >= 3 && degree(j)>=mean_item
                b = b + degree(j)/number_movies*sin(adj_matrix(i,j)*(y(e,j) - y(e,i)));
                b1 = b1 + degree(j)/number_movies*sin(adj_matrix(i,j)*(y(e-1,j) - y(e,i)));  
            end
        end
        if i<=number_user
            y(e+1,i)=K1*(degree(i)/number_user)*y(e,i)*(1-y(e,i))+K2*(a+sigma*a1)+(1-K1-K2)*(b+sigma*b1);
        else
            y(e+1,i)=K1*(degree(i)/number_movies)*y(e,i)*(1-y(e,i))+K2*(a+sigma*a1)+(1-K1-K2)*(b+sigma*b1);
        end
   end
   if norm(y(e+1,:)-y(e,i))<1e-3  
       break
   end
end
%% Predict score
if isnan(y(e+1,1:number_user))
    MAE=0;
    RMSE=0;
else
    b=pdist(y(e+1,1:number_user)');
    b1=linkage(b);
    a=cluster(b1,Ucluster_number);
    for i=1:max(a)
        temp=find(a==i);
        data=score_matrix(temp,:);    
        dex{i}=temp; 
        switch lower(method)
            case 'cosine'
                sim_matrix{i}=SimilitudItems(data','cosine');
            case 'correlation'
                sim_matrix{i}=SimilitudItems(data','correlation'); 
            case 'adjustedcosine'
                sim_matrix{i}=SimilitudItems(data','adjustedcosine');
            case 'jaccard'
                sim_matrix{i}=SimilitudItems(data','jaccard');
            case 'jaccardmsd'
                sim_matrix{i}=SimilitudItems(data','jaccardmsd');
        end
        sim_matrix{i}=sim_matrix{i}./repmat(sqrt(sum(sim_matrix{i}.^2,2)),1,size(sim_matrix{i},2));
    end
    if mycase==1
        load new\u1test
    elseif mycase==2
        load new\u2test
    elseif mycase==3
        load new\u3test
    elseif mycase==4
        load new\u4test
    elseif mycase==5
        load new\u5test
    end
    pp=find(u2(:,2)>max(u1(:,2)));
    [m,temp]=size(u2);
    Predict_score=zeros(m,1);
    for i=1:size(pp,1)
        user=u2(pp(i),1);
        [temp,BB]=find(score_matrix(user,:)~=0); 
        aver_score=mean(score_matrix(user,BB));
        Predict_score(pp(i))=round(aver_score);
    end
    for i=1:m
        if ismember(i,setdiff([1:m],pp))
            user=u2(i,1);
            item=u2(i,2);
            no2=a(user);
            user1=find(dex{no2}==user);
            up_score_matrix=score_matrix(dex{no2},:);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              [temp,BB]=find(up_score_matrix(user1,:)~=0);    
            aver_score=mean(up_score_matrix(user1,BB)); 
            P_u=find(up_score_matrix(:,item)~=0); 
            if isempty(P_u)
                Predict_score(i)=round(aver_score);
            else
                P_u_sim=sim_matrix{no2}(user1,P_u);  
                [temp,index1]=sort(P_u_sim,2,'descend');
                num1=size(index1,2);
                if num1>=neibor_num
                    neibor=(P_u(index1(1:neibor_num)));
                else
                    neibor=(P_u(index1));
                end
                sum1=0;
                sum2=0;
                for j=1:size(neibor,1)
                    [temp,BB]=find(up_score_matrix(neibor(j),:)~=0);
                    a_score(j)=mean(up_score_matrix(neibor(j),BB));
                    sum1 = sum1+sim_matrix{no2}(user1,neibor(j))*(up_score_matrix(neibor(j),item)-a_score(j));
                    sum2 = sum2+sim_matrix{no2}(user1,neibor(j));
                end
                if sum2==0   
                    Predict_score(i)=round(aver_score); 
                else
                    Predict_score(i)=round(aver_score+sum1/sum2);
                end
             end
            no_score=find(sum(score_matrix,2)==0);
            for w1=1:size(no_score,1)
                no_id=find(u2(:,1)==no_score(w1,1));
                for w2=1:size(no_id,1)
                    Predict_score(no_id(w2))=3;
                end
            end
            if Predict_score(i)>5
                Predict_score(i)=5;
            elseif Predict_score(i)<1
                Predict_score(i)=1;
            elseif isnan(Predict_score(i))
                Predict_score(i)=round(aver_score);
            end
        end
    end
    %% compute MAE and RMSE
    Eval=abs(u2(:,3)-Predict_score);
    RMSE=sqrt(Eval'*Eval/m);
    MAE=sum(Eval)/m; 
end