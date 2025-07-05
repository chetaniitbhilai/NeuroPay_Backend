// Import Product model
import Product from '../models/product.model.js';

// GET: Fetch all products
export const getAllProducts = async (req, res) => {
  try {
    const products = await Product.find();
    res.status(200).json(products);
  } catch (err) {
    res.status(500).json({
      error: 'Failed to fetch products',
      details: err.message,
    });
  }
};

// POST: Create a new product
export const createProduct = async (req, res) => {
  try {
    const { name, price, category, image } = req.body;

    // Basic validation
    if (!name || !price || !category || !image) {
      return res.status(400).json({
        error: 'All fields are required: name, price, category, image',
      });
    }

    const newProduct = new Product({ name, price, category, image });
    const savedProduct = await newProduct.save();

    res.status(201).json(savedProduct);
  } catch (err) {
    res.status(500).json({
      error: 'Failed to create product',
      details: err.message,
    });
  }
};
