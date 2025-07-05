import Payment from '../models/payment.model.js';
import dotenv from 'dotenv';
import Stripe from 'stripe';

dotenv.config();
const stripe = new Stripe(process.env.STRIPE_SECRET_KEY);

export const createPaymentIntent = async (req, res) => {
  const { totalAmount, items } = req.body;
  const userId = req.user?.id;

  console.log('Payment Request:', req.body);

  if (!userId || !totalAmount || !items || !Array.isArray(items)) {
    return res.status(400).json({ error: 'Invalid request body' });
  }

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
      amount: Math.round(totalAmount * 100), // amount in paise
      currency: 'inr',
      payment_method_types: ['card'],
      metadata: { userId },
    });



    await Payment.create({
      userId,
      amount: totalAmount,
      items: processedItems,
      paymentIntentId: paymentIntent.id,
      status: paymentIntent.status,
      automatic_payment_methods: { enabled: true }, 
      date: new Date(), // ✅ capture timestamp
    });

    console.log('Payment Intent Created:', paymentIntent.client_secret);
    
    res.status(200).json({ clientSecret: paymentIntent.client_secret });
  } catch (err) {
    console.error('Stripe error:', err);
    res.status(500).json({ error: 'Payment failed' });
  }
};
