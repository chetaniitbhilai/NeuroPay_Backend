import User from "../models/user.model.js";
import bcryptjs from "bcryptjs";
import generateTokenAndSetCookie from "../utils/generateToken.js";


export const signup = async (req, res) => {
  try {
    const {
      name,
      email,
      password,
      confirmPassword,
      address: { street, city, country, pincode }
    } = req.body;

    // Basic validation
    if (!name || !email || !password || !confirmPassword || !street || !city || !country || !pincode) {
      return res.status(400).json({ error: "All fields are required" });
    }

    if (password !== confirmPassword) {
      return res.status(400).json({ error: "Passwords do not match" });
    }

    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ error: "User already exists" });
    }

    // Hash password
    const salt = await bcryptjs.genSalt(10);
    const hashedPassword = await bcryptjs.hash(password, salt);

    // Create and save user
    const newUser = new User({
      name,
      email,
      password: hashedPassword,
      address: { street, city, country, pincode }
    });

    await newUser.save();

    res.status(201).json({ message: "User registered successfully" });

  } catch (error) {
    console.error("Signup Error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
};


export const login = async (req, res) => {
  try {
    const { email, password } = req.body;
    const user = await User.findOne({ email });

    if (!user) return res.status(400).json({ error: "Invalid credentials" });

    const isMatch = await bcryptjs.compare(password, user.password);
    if (!isMatch) return res.status(400).json({ error: "Invalid credentials" });
    

    const token = generateTokenAndSetCookie(user._id, res);
    res.status(200).json({
      _id: user._id,
      email: user.email,
      name: user.name,
      token,
    });
    console.log("User found:", user);
  } catch (error) {
    console.log("Error in login controller", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
};


export const logout = async(req, res) => {
    try {
        res.cookie("jwt","",{maxAge:0}); // cleared the cookie
        res.status(200).json({message:"Logged out successfully"});
    } catch (error) {
        console.log("Error in signup controller", error.message);
        res.status(500).json({ error: "Internal server error" })
    }
}

// controller
export const profile = async (req, res) => {
  try {
    const userId = req.user.id; // ✅ fixed
    console.log(userId);
    const user = await User.findById(userId);
    if (!user) return res.status(404).json({ error: 'User not found' });

    const response = {
      _id: user._id,
      email: user.email,
      name: user.name,
      address: user.address,
    };

    res.status(200).json(response);
  } catch (error) {
    console.error('Profile error:', error.message);
    res.status(500).json({ error: 'Internal server error' });
  }
};

