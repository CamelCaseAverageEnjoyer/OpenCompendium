mport turtle

rate = 20
x_box = 25
y_box = 25

turtle.speed(0)  # Fastest speed
turtle.pensize(1)
for i in range(-x_box, x_box + 1):
    turtle.penup()
    turtle.goto(i * rate, (- y_box) * rate)
    turtle.pendown()
    turtle.goto(i * rate, y_box * rate)
for i in range(-y_box, y_box + 1):
    turtle.penup()
    turtle.goto((- x_box) * rate, i * rate)
    turtle.pendown()
    turtle.goto(x_box * rate, i * rate)
turtle.penup()
turtle.goto(0, 0)
turtle.pensize(2)
turtle.pencolor("blue")
turtle.pendown()

# Изначально черепаха направлена направо, так что надо развернуть её вверх
turtle.left(90)

# Потом делаем то, что от нас просят
turtle.right(315)
for i in range(0, 7):
    turtle.forward(16 * rate)
    turtle.right(45)
    turtle.forward(8 * rate)
    turtle.right(135)

# Господи, что от вас хотят!

# Чтобы экран не закрывался
turtle.done()
