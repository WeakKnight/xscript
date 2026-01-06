// XScript Hello World Example
// Demonstrates basic syntax and features

// Variable declaration
var message = "Hello, XScript!";
print(message);

// Basic arithmetic
var x = 10;
var y = 20;
var sum = x + y;
print("Sum: " + sum);

// Function definition
func greet(name) {
    return "Hello, " + name + "!";
}

print(greet("World"));

// Control flow
var score = 85;

if (score >= 90) {
    print("Grade: A");
} else if (score >= 80) {
    print("Grade: B");
} else if (score >= 70) {
    print("Grade: C");
} else {
    print("Grade: F");
}

// Loops
print("Counting to 5:");
for (var i = 1; i <= 5; i += 1) {
    print(i);
}

// Tables (objects/arrays)
var player = {
    name: "Hero",
    hp: 100,
    mp: 50,
    level: 1
};

print("Player: " + player.name);
print("HP: " + player.hp);

// Anonymous functions
var double = func(n) {
    return n * 2;
};

print("Double of 21: " + double(21));

print("Done!");

