// XScript NPC Behavior Example
// Demonstrates game scripting patterns

// ============================================================
// NPC State Machine
// ============================================================

var STATE_IDLE = 0;
var STATE_PATROL = 1;
var STATE_CHASE = 2;
var STATE_ATTACK = 3;
var STATE_FLEE = 4;

// NPC constructor
func create_npc(name, x, y) {
    var npc = {
        name: name,
        x: x,
        y: y,
        state: STATE_IDLE,
        target: nil,
        hp: 100,
        max_hp: 100,
        damage: 10,
        speed: 2.0,
        sight_range: 10.0,
        attack_range: 1.5,
        patrol_points: {},
        patrol_index: 0
    };
    return npc;
}

// Distance calculation
func distance(x1, y1, x2, y2) {
    var dx = x2 - x1;
    var dy = y2 - y1;
    return (dx * dx + dy * dy) ^ 0.5;
}

// ============================================================
// State Behaviors
// ============================================================

func npc_idle(npc, dt) {
    // Check for nearby enemies
    var player = find_player();
    if (player != nil) {
        var dist = distance(npc.x, npc.y, player.x, player.y);
        if (dist < npc.sight_range) {
            npc.target = player;
            npc.state = STATE_CHASE;
            print(npc.name + " spotted the player!");
        }
    }
}

func npc_patrol(npc, dt) {
    // Move to next patrol point
    if (npc.patrol_points[npc.patrol_index] != nil) {
        var point = npc.patrol_points[npc.patrol_index];
        var dist = distance(npc.x, npc.y, point.x, point.y);
        
        if (dist < 0.5) {
            // Reached point, go to next
            npc.patrol_index = (npc.patrol_index + 1) % #npc.patrol_points;
        } else {
            // Move towards point
            move_towards(npc, point.x, point.y, dt);
        }
    }
    
    // Check for player
    var player = find_player();
    if (player != nil) {
        var dist = distance(npc.x, npc.y, player.x, player.y);
        if (dist < npc.sight_range) {
            npc.target = player;
            npc.state = STATE_CHASE;
        }
    }
}

func npc_chase(npc, dt) {
    if (npc.target == nil) {
        npc.state = STATE_IDLE;
        return;
    }
    
    var dist = distance(npc.x, npc.y, npc.target.x, npc.target.y);
    
    // Check if target is out of range
    if (dist > npc.sight_range * 1.5) {
        print(npc.name + " lost sight of target");
        npc.target = nil;
        npc.state = STATE_IDLE;
        return;
    }
    
    // In attack range?
    if (dist < npc.attack_range) {
        npc.state = STATE_ATTACK;
        return;
    }
    
    // Move towards target
    move_towards(npc, npc.target.x, npc.target.y, dt);
}

func npc_attack(npc, dt) {
    if (npc.target == nil) {
        npc.state = STATE_IDLE;
        return;
    }
    
    var dist = distance(npc.x, npc.y, npc.target.x, npc.target.y);
    
    // Out of attack range?
    if (dist > npc.attack_range) {
        npc.state = STATE_CHASE;
        return;
    }
    
    // Attack!
    deal_damage(npc.target, npc.damage);
    print(npc.name + " attacks for " + npc.damage + " damage!");
}

func npc_flee(npc, dt) {
    if (npc.target != nil) {
        // Move away from target
        var dx = npc.x - npc.target.x;
        var dy = npc.y - npc.target.y;
        var dist = distance(0, 0, dx, dy);
        
        if (dist > 0) {
            npc.x = npc.x + (dx / dist) * npc.speed * dt;
            npc.y = npc.y + (dy / dist) * npc.speed * dt;
        }
    }
    
    // Check if far enough
    if (npc.target == nil or 
        distance(npc.x, npc.y, npc.target.x, npc.target.y) > npc.sight_range * 2) {
        npc.state = STATE_IDLE;
        npc.target = nil;
    }
}

// ============================================================
// Helper Functions
// ============================================================

func move_towards(npc, tx, ty, dt) {
    var dx = tx - npc.x;
    var dy = ty - npc.y;
    var dist = distance(0, 0, dx, dy);
    
    if (dist > 0) {
        npc.x = npc.x + (dx / dist) * npc.speed * dt;
        npc.y = npc.y + (dy / dist) * npc.speed * dt;
    }
}

func on_damage(npc, amount) {
    npc.hp = npc.hp - amount;
    print(npc.name + " takes " + amount + " damage! HP: " + npc.hp);
    
    if (npc.hp <= 0) {
        print(npc.name + " has been defeated!");
        return true;  // Dead
    }
    
    // Low health? Flee!
    if (npc.hp < npc.max_hp * 0.2) {
        npc.state = STATE_FLEE;
    }
    
    return false;  // Still alive
}

// ============================================================
// Main Update Function
// ============================================================

func npc_update(npc, dt) {
    if (npc.state == STATE_IDLE) {
        npc_idle(npc, dt);
    } else if (npc.state == STATE_PATROL) {
        npc_patrol(npc, dt);
    } else if (npc.state == STATE_CHASE) {
        npc_chase(npc, dt);
    } else if (npc.state == STATE_ATTACK) {
        npc_attack(npc, dt);
    } else if (npc.state == STATE_FLEE) {
        npc_flee(npc, dt);
    }
}

// ============================================================
// Test
// ============================================================

print("NPC Behavior System Loaded");

// Create test NPC
var goblin = create_npc("Goblin", 0, 0);
print("Created NPC: " + goblin.name);
print("Initial state: " + goblin.state);

