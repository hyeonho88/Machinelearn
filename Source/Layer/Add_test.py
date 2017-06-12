import AddLayer as AL
import MulLayer as ML

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

mul_apple_layer = ML.MulLayer()
mul_orange_layer = ML.MulLayer()
add_apple_orange_layer = AL.AddLayer()
mul_tax_layer = ML.MulLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple,apple_num)
orange_price = mul_orange_layer.forward(orange,orange_num)
all_price = add_apple_orange_layer.forward(apple_price,orange_price)
price = mul_tax_layer.forward(all_price,tax)

print(price)

# 역전파

dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price,dorange_price = add_apple_orange_layer.backward(dall_price)
dorange,dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple_num,dapple, dorange, dorange_num, dtax)
