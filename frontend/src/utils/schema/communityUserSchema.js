import mongoose from 'mongoose';

const communityUserSchema = mongoose.Schema({   
    fullName: String,
    providerType: String,
    orgName: String,
    email: String,
    phone: String,
    createdAt: Date,
    // documentType: '',
})

const CommunityUser = mongoose.models.communityUser || mongoose.model('communityUser', communityUserSchema);

export default CommunityUser;