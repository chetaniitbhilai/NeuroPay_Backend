import mongoose from 'mongoose';

const paymentSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
  amount: Number,
  items: Array,
  paymentIntentId: String,  // for Stripe or Razorpay
  sessionId: String,        // for Stripe Checkout
  status: { type: String, default: 'created' }, // created, succeeded, failed, etc.
  method: { type: String, default: 'card' },    // 'card', 'upi', 'wallet', etc.
  vpa: { type: String, default: '' },           // UPI VPA (Virtual Payment Address) if available
  location: { type: String, default: 'unknown' },   // user or merchant location
  merchantId: { type: String, default: 'Walmart India' }, // optional merchant/business name
  date: { type: Date, default: Date.now },  // payment timestamp
  
  // Fraud detection fields
  fraudDetection: {
    isChecked: { type: Boolean, default: false },
    fraudProbability: { type: Number, default: 0.0 },
    prediction: { type: Number, default: 0 }, // 0 = legitimate, 1 = fraud
    riskLevel: { type: String, enum: ['LOW', 'MEDIUM', 'HIGH'], default: 'LOW' },
    isFraud: { type: Boolean, default: false },
    modelUsed: { type: String, default: 'rgtan' },
    checkedAt: { type: Date }
  }
});

export default mongoose.model('Payment', paymentSchema);
