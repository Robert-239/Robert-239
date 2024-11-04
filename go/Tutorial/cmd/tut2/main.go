package main

import ("fmt";
	"unicode/utf8"
)

func main()  {
	var intNum int = 32767

	fmt.Println(intNum)

	var floatNum float64 = 12345678.9
	fmt.Println(floatNum)

	var floatNum32 float32 = 10.1
	var intNum32 int32 = 2
	var result  float32 = floatNum32 + float32(intNum32)

	fmt.Println(result)

	var myString string = "Hello" + " " + "World"
	fmt.Println(myString)

	fmt.Println(utf8.RuneCountInString("Y"))

	var myRune rune = 'a'
	fmt.Println(myRune)

}
