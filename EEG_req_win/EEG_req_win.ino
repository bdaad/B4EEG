// 変数.
int analog_value_x = 0;
int analog_value_y = 0;
int analog_value_z = 0;

void setup() {
  Serial.begin(115200);  // シリアル通信開始
  pinMode(A6, INPUT);
  pinMode(A7, INPUT);
  pinMode(A12, INPUT);
}

int test_signal = 0;

void loop() {
  // PCからのリクエストを受信
  if (Serial.available()) {
    String request = Serial.readStringUntil('\n');  // 改行まで読み込み
    if (request == "req") {
      // アナログ値の読み取り
      analog_value_x = analogRead(A6);
      analog_value_y = analogRead(A7);
      analog_value_z = analogRead(A12);
      test_signal = test_signal + 1;
      if (test_signal > 5000){
        test_signal = 0;
      }
      analog_value_x = test_signal;


      // センサー値をPCに送信
      Serial.print(analog_value_x);
      Serial.print(',');
      Serial.print(analog_value_y);
      Serial.print(',');
      Serial.println(analog_value_z);
    }
  }
}
