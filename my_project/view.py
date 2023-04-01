from django.shortcuts import render
import pandas as pd 
from . import finalproject
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def predict(request):
    return render(request,'predict.html')    

def source(request):
    return render(request,'source.html')


def result(request):
    result1=""
    if len(request.POST.dict())!=8: return  render(request,'result.html',{"result2":"Incomplete request"})  
    val1=float(request.POST['n1'])
    val2=float(request.POST['n2']) 
    val3=float(request.POST['n3'])
    val4=float(request.POST['n4'])
    val5=float(request.POST['n5'])
    val6=float(request.POST['n6'])
    val7=float(request.POST['n7'])
    val8=float(request.POST['n8'])
    
    prediction=finalproject.prediction_with_random_forest([[val1,val2,val3,val4,val5,val6,val7,val8]])

    if prediction:
        result1="Your chances of being diabetic is Positive"
    else : result1="Your chances of being diabetic is Negative"
    print(prediction,result1)
    return  render(request,'result.html',{"result2":result1})  
    # return HttpResponse({result1}) 