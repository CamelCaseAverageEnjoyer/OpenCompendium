void setup() {
  // Устанавливаем пин светодиода в режим выхода.
  // Используем определение LED_BUILTIN,
  // которое содержит в себе пин светодиода
  pinMode(LED_BUILTIN, OUTPUT);
}
 
void loop() {
  // Включаем светодиод
  digitalWrite(LED_BUILTIN, HIGH);
  // Ждём пол секунды
  delay(500);
  // Выключаем светодиод                       
  digitalWrite(LED_BUILTIN, LOW);
  // Ждём пол секунды   
  delay(1000);                       
}