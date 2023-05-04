mycase=[1,2,3,4,5];
neibor_num=[10,20,30,40,50,60,70,80,90,100];
ml_MAE=zeros(5,10);
ml_RMSE=zeros(5,10);
for i=1:size(mycase,2)
    for j=1:size(neibor_num,2)
         [MAE,RMSE]=EC_LIUP(mycase(i),neibor_num(j),'jaccardmsd');
         ml_MAE(i,j)=MAE;
         ml_RMSE(i,j)=RMSE;
    end 
end
