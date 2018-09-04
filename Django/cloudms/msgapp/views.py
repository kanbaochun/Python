from django.shortcuts import render
from datetime import datetime
from django.http import HttpResponse, JsonResponse
# Create your views here.

def msgproc(request):
    datalist =[]
    if request.method == "POST":
        userA = request.POST.get("userA", None)
        userB = request.POST.get("userB", None)
        msg = request.POST.get("msg", None)
        time = datetime.now()
        with open('msgdata.txt', 'a+') as f:
            f.write("{}--{}--{}--{}--\n".format(userA, userB,
                        msg, time.strftime("%Y-%m-%d %H:%M%S")))

    if request.method == "GET":
        userC = request.GET.get("userC", None)
        if userC != None:
            with open('msgdata.txt', 'r') as f:
                cnt = 0
                for line in f:
                    linedata = line.split('--')
                    if linedata[0] == userC:
                        cnt = cnt + 1
                        d = {"userA":linedata[1], "msg":linedata[2],
                             "time":linedata[3]}
                        datalist.append(d)
                    if cnt >= 10:
                        break
    return render(request, "msgSingleWeb.html", {"data":datalist})

def homeproc(request):
    response = HttpResponse()
    response.write("<h1>这是首页，具体功能请访问<a href='./msggate'>这里</a></h1>")
    response.write("<h1>这里是第二行</h1>")
    return response
    #return HttpResponse("<h1>这是首页，具体功能请访问<a href='./msggate'>这里</a></h1>")

def homeproc1(request):
    response = JsonResponse({'key':'value'})
    return response

def homepro2(request):
    pass