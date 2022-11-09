function setClock(){
    var dateInfo = new Date();
    var hour = modifyNumber(dateInfo.getHours());
    var min = modifyNumber(dateInfo.getMinutes());
    var sec = modifyNumber(dateInfo.getSeconds());
    var time=610;
    v_min=min%10;
    if(v_min==0 && sec<=10){
        v_min=10;
    }
    time=time-(v_min*60)-sec;
    min1=modifyNumber(parseInt(time/60));
    sec1=modifyNumber(time%60);

    document.getElementById("time1").innerHTML = min1  + ":" + sec1;
    document.getElementById("date1").innerHTML = "Until Update"
    time--;
    if(time<0){
        location.reload();
    }
}
function modifyNumber(time){
    if(parseInt(time)<10){
        return "0"+ time;
    }
    else
        return time;
}

setClock();
setInterval(setClock,1000); //1초마다 setClock 함수 실행
