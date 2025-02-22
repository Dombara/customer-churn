import mongoose from 'mongoose';

const ResourceSchema = new mongoose.Schema({
  location: String,
  category: String,
  capacity: String,
  contactInfo: String,
  operatingHours: String,
  // specialServices:String,
  notes: String,
  createdAt: Date,
});

const Resource = mongoose.models.Resource || mongoose.model('Resource', ResourceSchema);

export default Resource;
