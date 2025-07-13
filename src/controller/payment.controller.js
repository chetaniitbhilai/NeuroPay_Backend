import dotenv from 'dotenv';
import Stripe from 'stripe';
import Payment from '../models/payment.model.js';
import axios from 'axios';

dotenv.config();
const stripe = new Stripe(process.env.STRIPE_SECRET_KEY);

async function checkFraud(paymentData) {
  try {
    const response = await axios.post('http://localhost:5004/predict', paymentData);
    return response.data;

  } catch (error) {
    console.error('Error during fraud detection:', error);
    return null;
  }
}

async function getUserFraudSummary(userId) {
  try {
    const response = await axios.get(`http://localhost:5001/api/user-fraud-summary`, {
      params: { user_id: userId }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching user fraud summary:', error);
    return null;
  }
}

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

    // Check for fraud
    const now = new Date().toISOString();
const fraudItem = {
      Time: now,
      Source: userId,
      Target: merchantId,
      Amount: totalAmount,
      Location: location,
      Type: 'card',
    };
    
    
    const fraudResult = await checkFraud(fraudItem);
    const gesResult = await getUserFraudSummary(userId);
    const fraud_probability = (fraudResult?.fraud_probability + (gesResult?.fraud_true/(gesResult?.fraud_true + gesResult?.fraud_false))) / 2 || 0.0;
    let riskLevel = 'LOW';
    if (fraud_probability > 0.5) {
      riskLevel = 'MEDIUM';
    } else if (fraud_probability >0.7) {
       riskLevel = 'HIGH';
    } else {
      riskLevel = 'LOW';
    }

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

      // Store fraud detection results
      fraudDetection: {
        isChecked: true,
        fraudProbability: fraud_probability || 0.0,
        prediction: fraudResult?.prediction || 0,
        riskLevel: riskLevel || 'LOW',
        isFraud: fraudResult?.is_fraud || false,
        modelUsed: fraudResult?.model_used || 'rgtan',
        checkedAt: new Date()
      }
    });
    // Log the successful creation
    console.log('✅ Payment Intent Saved to DB:', paymentIntent.id);
    console.log('✅ Payment Intent Created:', paymentIntent.client_secret);

    res.status(200).json({ clientSecret: paymentIntent.client_secret, paymentIntentId: paymentIntent.id, fraudResult });
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

    // Check for fraud (UPI)
    const nowUpi = new Date().toISOString();
const fraudItemUpi = {
      Time: nowUpi,
      Source: userId,
      Target: merchantId,
      Amount: totalAmount,
      Location: location,
      Type: 'upi',
    };
    const fraudResult = await checkFraud(fraudItemUpi);
    const gesResult = await getUserFraudSummary(userId);
    const fraud_probability = (fraudResult?.fraud_probability + (gesResult?.fraud_true/(gesResult?.fraud_true + gesResult?.fraud_false))) / 2 || 0.0;
    let riskLevel = 'LOW';
    if (fraud_probability > 0.7) {
      riskLevel = 'HIGH';
    } else if (fraud_probability > 0.5) {
      riskLevel = 'MEDIUM';
    } else {
      riskLevel = 'LOW';
    }
    console.log('UPI Payment Fraud Result:', fraud_probability, riskLevel);

    const paymentRecord = new Payment({
      userId,
      amount: totalAmount,
      items: processedItems,
      vpa,
      method: 'upi',
      status: 'succeeded',
      location,
      merchantId,

      // Store fraud detection results
      fraudDetection: {
        isChecked: true,
        fraudProbability: fraud_probability || 0.0,
        prediction: fraudResult?.prediction || 0,
        riskLevel: riskLevel || 'LOW',
        isFraud: fraudResult?.is_fraud || false,
        modelUsed: fraudResult?.model_used || 'rgtan',
        checkedAt: new Date()
      }
    });

    await paymentRecord.save();

    res.status(200).json({ message: 'UPI payment logged successfully', fraudResult });
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
