package com.example.administrator.sentio_web;
import android.Manifest;
import android.content.Intent;
import android.os.Build;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import android.view.View;

import java.io.IOException;
import java.util.ArrayList;

import android.os.Handler;
import android.os.Message;

import android.Manifest;
import android.content.Intent;
import android.os.Build;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import java.util.ArrayList;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.SocketAddress;
import java.net.UnknownHostException;

import android.speech.tts.TextToSpeech;
import static android.speech.tts.TextToSpeech.ERROR;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {

    private TextToSpeech tts;

    Intent intent;
    SpeechRecognizer mRecognizer;
    Button sttBtn;
    Button netBtn;

    TextView textView;
    final int PERMISSION = 1;

    ////////////////////
    // 통신
    //////////////////
    public TextView tv;

    public Socket cSocket = null;
    private String server = "192.168.0.9";  // 서버 ip주소
    private int port = 9797;                           // 포트번호

    public DataInputStream ip = null;
    public PrintWriter streamOut = null;
    public BufferedReader streamIn = null;

    public chatThread cThread = null;
    public Thread log_Thread = null;

    public String message = "";

    private void sendMessage(String MSG) {
        try {
            if(streamOut != null){
                streamOut.println(MSG);     // 서버에 메세지를 보내줍니다.
            }else{
                logger("Null streamOut!!!!");
            }

        } catch (Exception ex) {
            logger("Error:" + ex.toString());
            logger(ex.toString());
        }

    }
    /////////////////

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if ( Build.VERSION.SDK_INT >= 23 ){
            // 퍼미션 체크
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.INTERNET,
                    Manifest.permission.RECORD_AUDIO},PERMISSION);
        }

        textView = (TextView)findViewById(R.id.sttResult);
        tv = (TextView)findViewById(R.id.tv2);
        sttBtn = (Button) findViewById(R.id.sttStart);

        intent=new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        intent.putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE,getPackageName());
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE,"ko-KR");
        sttBtn.setOnClickListener(v ->{
            mRecognizer=SpeechRecognizer.createSpeechRecognizer(this);
            mRecognizer.setRecognitionListener(listener);
            mRecognizer.startListening(intent);
        });

        netBtn = (Button) findViewById(R.id.connBtn);
        netBtn.setOnClickListener(v ->{
            if (cSocket == null) {
                logger("접속중입니다...");

                new Thread() {
                    public void run() {
                        //connect(server, port , nickName);

                        try{
                            SocketAddress socketAddress = new InetSocketAddress(server, port);

                            logger( " connect Start " + server + " " + port);

                            cSocket = new Socket(server, port);
                            cSocket.setSoTimeout(50000/* 타임 아웃 시간 ms단위 */);

                            //cSocket.connect(socketAddress, 5000/* 타임 아웃 시간 ms단위 */);

                            streamOut = new PrintWriter(cSocket.getOutputStream(), true);      // 출력용 스트림
                            ip = new DataInputStream(cSocket.getInputStream());
                            streamIn = new BufferedReader(new InputStreamReader(ip));  // 입력용 스트림
                            cThread = new chatThread(cSocket);
                            cThread.start();
                            sendMessage("hi");

                            sendMessage("hi2");

                        }catch(Exception ex){
                            textView.setText("error!" + ex.getMessage());
                        }
                    }
                }.start();

            }
        });

        tts = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if(status != ERROR) {
                    // 언어를 선택한다.
                    tts.setLanguage(Locale.KOREAN);
                }
            }
        });

        tts.speak("안녕하세요",TextToSpeech.QUEUE_FLUSH, null);

    }

    private RecognitionListener listener = new RecognitionListener() {
        @Override
        public void onReadyForSpeech(Bundle params) {
            Toast.makeText(getApplicationContext(),"음성인식을 시작합니다.",Toast.LENGTH_SHORT).show();
        }

        @Override
        public void onBeginningOfSpeech() {}

        @Override
        public void onRmsChanged(float rmsdB) {}

        @Override
        public void onBufferReceived(byte[] buffer) {}

        @Override
        public void onEndOfSpeech() {}

        @Override
        public void onError(int error) {
            String message;

            switch (error) {
                case SpeechRecognizer.ERROR_AUDIO:
                    message = "오디오 에러";
                    break;
                case SpeechRecognizer.ERROR_CLIENT:
                    message = "클라이언트 에러";
                    break;
                case SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS:
                    message = "퍼미션 없음";
                    break;
                case SpeechRecognizer.ERROR_NETWORK:
                    message = "네트워크 에러";
                    break;
                case SpeechRecognizer.ERROR_NETWORK_TIMEOUT:
                    message = "네트웍 타임아웃";
                    break;
                case SpeechRecognizer.ERROR_NO_MATCH:
                    message = "찾을 수 없음";
                    break;
                case SpeechRecognizer.ERROR_RECOGNIZER_BUSY:
                    message = "RECOGNIZER가 바쁨";
                    break;
                case SpeechRecognizer.ERROR_SERVER:
                    message = "서버가 이상함";
                    break;
                case SpeechRecognizer.ERROR_SPEECH_TIMEOUT:
                    message = "말하는 시간초과";
                    break;
                default:
                    message = "알 수 없는 오류임";
                    break;
            }

            Toast.makeText(getApplicationContext(), "에러가 발생하였습니다. : " + message,Toast.LENGTH_SHORT).show();
        }

        @Override
        public void onResults(Bundle results) {
            // 말을 하면 ArrayList에 단어를 넣고 textView에 단어를 이어줍니다.
            ArrayList<String> matches =
                    results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);

            message = "";

            for(int i = 0; i < matches.size() ; i++){
                textView.setText(matches.get(i));
            }
            message = matches.get(matches.size() - 1);

            new Thread() {
                public void run() {
                    //connect(server, port , nickName);

                    try{
                        sendMessage(message);
                    }catch(Exception ex){
                        textView.setText("error!" + ex.getMessage());
                    }
                }
            }.start();


        }

        @Override
        public void onPartialResults(Bundle partialResults) {}

        @Override
        public void onEvent(int eventType, Bundle params) {}
    };

    private Thread checkUpdate = null;

    private void logger(String MSG) {
        tv.append(MSG + "\n");     // 텍스트뷰에 메세지를 더해줍니다.
        tts.speak(MSG,TextToSpeech.QUEUE_FLUSH, null);
    }

    public void connect(String server, int port, String user) {
        try {
            SocketAddress socketAddress = new InetSocketAddress(server, port);

            logger(" connect Start " + server + " " + port);

            cSocket = new Socket(server, port);
            cSocket.setSoTimeout(1000/* 타임 아웃 시간 ms단위 */);

            cSocket.connect(socketAddress, 3000/* 타임 아웃 시간 ms단위 */);


        }catch (UnknownHostException ex) {

        } catch (IOException ex) {

        }catch (Exception ex) {

        }
    }

    Handler mHandler = new Handler() {   // 스레드에서 메세지를 받을 핸들러.
        public void handleMessage(Message msg) {
            try{
                switch (msg.what) {
                    case 0: // 채팅 메세지를 받아온다.
                        logger(msg.obj.toString());
                        break;
                    case 1: // 소켓접속을 끊는다.
                        try {
                            cSocket.close();
                            cSocket = null;

                            logger("접속이 끊어졌습니다.");

                        } catch (Exception e) {
                            logger("접속이 이미 끊겨 있습니다." + e.getMessage());
                            finish();
                        }
                        break;
                }
            }catch (Exception e){
                logger(e.toString());
            }

        }
    };

    class chatThread extends Thread {
        private boolean flag = false; // 스레드 유지(종료)용 플래그
        public BufferedReader cTh_StreamIn = null;
        Socket Th_Socket = null;
        String myMsg = "";

        public chatThread(Socket socket){
            Th_Socket = socket;  // 입력용 스트림
            try{
                cTh_StreamIn = new BufferedReader(new InputStreamReader(new DataInputStream(Th_Socket.getInputStream())));
            }catch (Exception e){

            }
        }

        public void run() {
            try {
                while (!flag) { // 플래그가 false일경우에 루프

                    String msgs = "Loop Start";
                    Message msg = mHandler.obtainMessage();
                    msg.what = 0;
                    //msg.obj = msgs;

                    //mHandler.sendMessage(msg);

                    msgs = cTh_StreamIn.readLine();  // 서버에서 올 메세지를 기다린다.
                    msg.obj = msgs;

                    mHandler.sendMessage(msg); // 핸들러로 메세지 전송

                    if (msgs.equals("# [" +  "]님이 나가셨습니다.")) { // 서버에서 온 메세지가 종료 메세지라면
                        flag = true;   // 스레드 종료를 위해 플래그를 true로 바꿈.
                        msg = new Message();
                        msg.what = 1;   // 종료메세지
                        mHandler.sendMessage(msg);
                    }
                }

            }catch(Exception e) {
                //logger("Thread Error");
                //logger(e.getMessage());
                myMsg = e.toString();
                new Thread() {
                    public void run() {
                        streamOut.println("[" + "] " + myMsg);     // 서버에 메세지를 보내줍니다.
                    }
                }.start();
            }
        }
    };

}