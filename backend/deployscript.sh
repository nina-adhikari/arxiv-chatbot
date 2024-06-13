now="$(date +"%D")"

TAG="gcr.io/$PROJECT_ID/$APP/$now"
REGION="us-central1"
gcloud config set project $PROJECT_ID
docker build -t $TAG .
gcloud builds submit --tag $TAG
gcloud run deploy $APP --image $TAG --platform managed --region $REGION --allow-unauthenticated