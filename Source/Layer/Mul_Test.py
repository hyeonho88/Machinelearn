# coding : utf-8
import MulLayer as ML

apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = ML.MulLayer()
mul_tax_layer = ML.MulLayer()

apple_price = mul_apple_layer.forward(apple,apple_num)
price = mul_tax_layer.forward(apple_price,tax)

print("Forward :",price)

dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax)
