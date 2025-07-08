import dotenv from 'dotenv';
import Stripe from 'stripe';
import Payment from '../models/payment.model.js';

dotenv.config();
const stripe = new Stripe(process.env.STRIPE_SECRET_KEY);

export const createPaymentIntent = async (req, res) => {
  const { totalAmount, items, merchantId = 'Walmart India', location = 'India' } = req.body;
  const userId = req.user?.id;

  console.log('Incoming Payment Request:', req.body);

  if (!userId || !totalAmount || !Array.isArray(items)) {
    return res.status(400).json({ error: 'Invalid request body' });
  }

  // Process items to store in DB
  const processedItems = items.map(item => ({
    _id: item._id,
    name: item.name,
    price: item.price,
    quantity: item.quantity,
    total: item.price * item.quantity,
    image: item.image || '',
  }));

  try {
    const paymentIntent = await stripe.paymentIntents.create({
      amount: Math.round(totalAmount * 100), // Convert to paise
      currency: 'inr',
      payment_method_types: ['card'],
      metadata: { userId },
    });

    // Save payment record in MongoDB
    await Payment.create({
      userId,
      amount: totalAmount,
      items: processedItems,
      paymentIntentId: paymentIntent.id,
      status: paymentIntent.status,
      merchantId,
      location,
      date: new Date(),
    });
    // Log the successful creation
    console.log('✅ Payment Intent Saved to DB:', paymentIntent.id);
    console.log('✅ Payment Intent Created:', paymentIntent.client_secret);
    
    res.status(200).json({ clientSecret: paymentIntent.client_secret, paymentIntentId: paymentIntent.id });
  } catch (err) {
    console.error('❌ Stripe Error:', err);
    res.status(500).json({ error: 'Payment creation failed' });
  }
};


dotenv.config();

export const logUPIPayment = async (req, res) => {
  try {
    const { totalAmount, items, vpa, location = 'India', merchantId = 'Walmart India' } = req.body;
    const userId = req.user?.id;

    if (!userId || !totalAmount || !Array.isArray(items) || !vpa) {
      return res.status(400).json({ error: 'Missing required UPI payment data' });
    }

    const processedItems = items.map(item => ({
      _id: item._id,
      name: item.name,
      price: item.price,
      quantity: item.quantity,
      total: item.price * item.quantity,
      image: item.image || '',
    }));

    const paymentRecord = new Payment({
      userId,
      amount: totalAmount,
      items: processedItems,
      vpa,
      method: 'upi',
      status: 'succeeded',
      location,
      merchantId,
    });

    await paymentRecord.save();

    res.status(200).json({ message: 'UPI payment logged successfully' });
  } catch (error) {
    console.error('Error logging UPI payment:', error);
    res.status(500).json({ error: 'Internal server error while logging UPI payment' });
  }
};


export const markPaymentSuccess = async (req, res) => {
  const { paymentIntentId } = req.body;

  if (!paymentIntentId) {
    return res.status(400).json({ error: 'Missing paymentIntentId' });
  }

  try {
    const payment = await Payment.findOneAndUpdate(
      { paymentIntentId },
      { status: 'succeeded' },
      { new: true }
    );

    if (!payment) {
      return res.status(404).json({ error: 'Payment not found' });
    }

    return res.status(200).json({ message: 'Payment marked as succeeded', payment });
  } catch (err) {
    console.error('Error updating payment:', err);
    return res.status(500).json({ error: 'Failed to update payment status' });
  }
};


export const getUserSuccessfulPayments = async (req, res) => {
  try {
    const userId = req.user.id;

    const successfulPayments = await Payment.find({
      userId,
      status: 'succeeded',
    }).sort({ date: -1 });

    res.status(200).json(successfulPayments);
  } catch (err) {
    console.error('Error fetching payment history:', err);
    res.status(500).json({ error: 'Failed to fetch payment history' });
  }
};
