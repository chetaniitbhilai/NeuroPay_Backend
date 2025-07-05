import mongoose from 'mongoose';

const userSchema = new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },

  address: {
    street: { type: String, required: true },
    city: { type: String, required: true },
    country: { type: String, required: true },
    pincode: { type: String, required: true },
  },
}, {
  timestamps: true,
});

const User = mongoose.model('User', userSchema);
export default User;
