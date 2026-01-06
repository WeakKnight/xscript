// XScript Vector Math Example
// Demonstrates metatables and operator overloading

// ============================================================
// Vector3 Type with Metatable
// ============================================================

var Vector3 = {};

// Metatable for Vector3
var Vector3Meta = {
    // Addition
    __add: func(a, b) {
        return Vector3.new(a.x + b.x, a.y + b.y, a.z + b.z);
    },
    
    // Subtraction
    __sub: func(a, b) {
        return Vector3.new(a.x - b.x, a.y - b.y, a.z - b.z);
    },
    
    // Multiplication (scalar or dot product)
    __mul: func(a, b) {
        if (type(b) == "number") {
            return Vector3.new(a.x * b, a.y * b, a.z * b);
        } else if (type(a) == "number") {
            return Vector3.new(b.x * a, b.y * a, b.z * a);
        }
        // Dot product
        return a.x * b.x + a.y * b.y + a.z * b.z;
    },
    
    // Division by scalar
    __div: func(a, b) {
        return Vector3.new(a.x / b, a.y / b, a.z / b);
    },
    
    // Negation
    __neg: func(v) {
        return Vector3.new(-v.x, -v.y, -v.z);
    },
    
    // Equality
    __eq: func(a, b) {
        return a.x == b.x and a.y == b.y and a.z == b.z;
    },
    
    // String conversion
    __tostring: func(v) {
        return "(" + v.x + ", " + v.y + ", " + v.z + ")";
    }
};

// Make methods accessible via __index
Vector3Meta.__index = Vector3Meta;

// Constructor
Vector3.new = func(x, y, z) {
    var v = {
        x: x,
        y: y,
        z: z
    };
    setmetatable(v, Vector3Meta);
    return v;
};

// Zero vector
Vector3.zero = func() {
    return Vector3.new(0, 0, 0);
};

// Unit vectors
Vector3.right = func() {
    return Vector3.new(1, 0, 0);
};

Vector3.up = func() {
    return Vector3.new(0, 1, 0);
};

Vector3.forward = func() {
    return Vector3.new(0, 0, 1);
};

// ============================================================
// Vector Operations
// ============================================================

// Magnitude (length)
func vec_magnitude(v) {
    return (v.x * v.x + v.y * v.y + v.z * v.z) ^ 0.5;
}

// Squared magnitude (faster, no sqrt)
func vec_sqr_magnitude(v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

// Normalize
func vec_normalize(v) {
    var mag = vec_magnitude(v);
    if (mag > 0) {
        return v / mag;
    }
    return Vector3.zero();
}

// Dot product
func vec_dot(a, b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Cross product
func vec_cross(a, b) {
    return Vector3.new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// Distance between two points
func vec_distance(a, b) {
    return vec_magnitude(b - a);
}

// Linear interpolation
func vec_lerp(a, b, t) {
    return a + (b - a) * t;
}

// Reflect vector off a surface
func vec_reflect(v, normal) {
    return v - normal * 2 * vec_dot(v, normal);
}

// Project vector onto another
func vec_project(v, onto) {
    var dot = vec_dot(v, onto);
    var sqrMag = vec_sqr_magnitude(onto);
    return onto * (dot / sqrMag);
}

// Angle between vectors (in radians)
func vec_angle(a, b) {
    var dot = vec_dot(vec_normalize(a), vec_normalize(b));
    // Clamp to [-1, 1] to avoid NaN from acos
    if (dot > 1) { dot = 1; }
    if (dot < -1) { dot = -1; }
    // Note: Would need acos function
    return dot;  // Returns cos of angle for now
}

// ============================================================
// Test
// ============================================================

print("=== Vector3 Math Test ===");

var v1 = Vector3.new(1, 2, 3);
var v2 = Vector3.new(4, 5, 6);

print("v1 = " + tostring(v1));
print("v2 = " + tostring(v2));

var sum = v1 + v2;
print("v1 + v2 = " + tostring(sum));

var diff = v2 - v1;
print("v2 - v1 = " + tostring(diff));

var scaled = v1 * 2;
print("v1 * 2 = " + tostring(scaled));

var dot = v1 * v2;
print("v1 . v2 = " + dot);

var cross = vec_cross(v1, v2);
print("v1 x v2 = " + tostring(cross));

var mag = vec_magnitude(v1);
print("|v1| = " + mag);

var norm = vec_normalize(v1);
print("normalize(v1) = " + tostring(norm));

var lerped = vec_lerp(v1, v2, 0.5);
print("lerp(v1, v2, 0.5) = " + tostring(lerped));

print("=== Tests Complete ===");

